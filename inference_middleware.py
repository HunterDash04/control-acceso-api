"""
inference_middleware.py
------------------------
Módulo de INFERENCIA que usará el middleware.

Flujo:
    1. build_pipeline(pth_path)  -> se llama UNA sola vez al iniciar el
       middleware. Carga el .pth, reconstruye las arquitecturas y deja
       todo listo en memoria.
    2. process_batch(pipeline, images) -> se llama por cada lote de
       EXACTAMENTE 10 imágenes (frames de la ESP32-CAM) y devuelve el
       resultado del control de acceso.

Lógica implementada (según especificación):
    - Por cada una de las 10 imágenes se calcula:
        a) confianza de "humano" (detector SSDLite/MobileNetV3, clase COCO 'person')
        b) confianza/presencia de "rostro" (MTCNN) + embedding facial (FaceNet)
    - CASO A (alguna imagen >= umbral de humano):
        Top 5 ordenado primero por (tiene_rostro, confianza_humano) descendente.
        Se evalúa control de acceso sobre el rostro más claro del Top 5:
            * "ACCESO AUTORIZADO"     -> coincide con authorized_faces
            * "ACCESO NO AUTORIZADO"  -> hay rostro pero no coincide
            * "ACCESO SOSPECHOSO"     -> hay humano en el Top5 pero ningún
                                         rostro válido pudo extraerse
    - CASO B (ninguna imagen alcanza el umbral):
        Top 5 = las 5 de MENOR confianza. El sistema termina en silencio
        (silent=True): el middleware no debe reportar ni procesar accesos.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as TF

from facenet_pytorch import MTCNN, InceptionResnetV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_PERSON_CLASS_ID = 1
BATCH_SIZE_REQUIRED = 10
TOP_K = 5


@dataclass
class ImageResult:
    index: int
    human_confidence: float = 0.0
    face_confidence: float = 0.0   # probabilidad de detección del rostro (MTCNN)
    has_face: bool = False
    embedding: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------
# 1. CARGA DEL PIPELINE (una sola vez, al iniciar el middleware)
# --------------------------------------------------------------------------
def build_pipeline(pth_path: str = "modelo_control_acceso.pth") -> Dict[str, Any]:
    checkpoint = torch.load(pth_path, map_location=DEVICE)

    # --- Detector de humanos ---
    human_detector = ssdlite320_mobilenet_v3_large(
        weights=None,
        num_classes=checkpoint.get("human_detector_num_classes", 91),
    )
    human_detector.load_state_dict(checkpoint["state_dict"])
    human_detector.eval().to(DEVICE)

    # --- Pipeline facial ---
    mtcnn = MTCNN(
        image_size=160, margin=20, keep_all=False,
        post_process=True, device=DEVICE,
    )

    resnet = InceptionResnetV1(pretrained=None, classify=False)
    resnet.load_state_dict(checkpoint["facenet_state_dict"])
    resnet.eval().to(DEVICE)

    return {
        "human_detector": human_detector,
        "mtcnn": mtcnn,
        "resnet": resnet,
        "authorized_faces": checkpoint["authorized_faces"],
        "human_conf_threshold": checkpoint.get("human_conf_threshold", 0.60),
        "face_match_threshold": checkpoint.get("face_match_threshold", 0.70),
    }


# --------------------------------------------------------------------------
# 2. DETECCIÓN POR IMAGEN
# --------------------------------------------------------------------------
def _human_confidence(pipeline, image: Image.Image) -> float:
    """Devuelve la confianza (0-1) de la mejor detección de clase 'person'."""
    tensor = TF.to_tensor(image).to(DEVICE)
    with torch.no_grad():
        output = pipeline["human_detector"]([tensor])[0]

    best_score = 0.0
    for label, score in zip(output["labels"], output["scores"]):
        if int(label) == COCO_PERSON_CLASS_ID:
            best_score = max(best_score, float(score))
    return best_score


def _face_detection(pipeline, image: Image.Image):
    """Devuelve (face_confidence, embedding|None)."""
    mtcnn = pipeline["mtcnn"]

    # detect() entrega cajas + probabilidades sin recortar el rostro
    boxes, probs = mtcnn.detect(image)
    if boxes is None or probs is None or probs[0] is None:
        return 0.0, None

    face_confidence = float(probs[0])

    # mtcnn(image) entrega el tensor del rostro ya alineado para el embedding
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return face_confidence, None

    with torch.no_grad():
        embedding = pipeline["resnet"](face_tensor.unsqueeze(0).to(DEVICE))
    embedding = F.normalize(embedding.squeeze(0), p=2, dim=0).cpu()

    return face_confidence, embedding


def _analyze_image(pipeline, idx: int, image: Image.Image) -> ImageResult:
    human_conf = _human_confidence(pipeline, image)
    face_conf, embedding = _face_detection(pipeline, image)

    return ImageResult(
        index=idx,
        human_confidence=human_conf,
        face_confidence=face_conf,
        has_face=embedding is not None,
        embedding=embedding,
    )


# --------------------------------------------------------------------------
# 3. LÓGICA DE FILTRADO TOP-5 (Caso A / Caso B)
# --------------------------------------------------------------------------
def _select_top5(results: List[ImageResult], threshold: float):
    above_threshold = [r for r in results if r.human_confidence >= threshold]

    if above_threshold:
        # CASO A: primero las que tienen rostro, luego por confianza de humano
        ranked = sorted(
            above_threshold,
            key=lambda r: (r.has_face, r.human_confidence),
            reverse=True,
        )
        return ranked[:TOP_K], "A"

    # CASO B: nadie supera el umbral -> 5 con MENOR confianza, sistema silencioso
    ranked = sorted(results, key=lambda r: r.human_confidence)
    return ranked[:TOP_K], "B"


# --------------------------------------------------------------------------
# 4. LÓGICA DE CONTROL DE ACCESO (solo Caso A)
# --------------------------------------------------------------------------
def _match_identity(embedding: torch.Tensor, authorized_faces: dict, threshold: float):
    """Compara contra la base de autorizados usando similitud coseno."""
    best_name, best_sim = None, -1.0
    for name, ref_embedding in authorized_faces.items():
        sim = F.cosine_similarity(
            embedding.unsqueeze(0), ref_embedding.unsqueeze(0)
        ).item()
        if sim > best_sim:
            best_name, best_sim = name, sim

    if best_name is not None and best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


def _resolve_access(top5: List[ImageResult], pipeline) -> Dict[str, Any]:
    faces_in_top5 = [r for r in top5 if r.has_face]

    if not faces_in_top5:
        return {
            "status": "ACCESO SOSPECHOSO",
            "matched_identity": None,
            "similarity": None,
        }

    # imagen con el rostro más claro = mayor face_confidence
    clearest = max(faces_in_top5, key=lambda r: r.face_confidence)

    matched_name, similarity = _match_identity(
        clearest.embedding,
        pipeline["authorized_faces"],
        pipeline["face_match_threshold"],
    )

    if matched_name is not None:
        return {
            "status": "ACCESO AUTORIZADO",
            "matched_identity": matched_name,
            "similarity": round(similarity, 4),
        }

    return {
        "status": "ACCESO NO AUTORIZADO",
        "matched_identity": None,
        "similarity": round(similarity, 4) if similarity is not None else None,
    }


# --------------------------------------------------------------------------
# 5. FUNCIÓN PRINCIPAL QUE INVOCA EL MIDDLEWARE
# --------------------------------------------------------------------------
def process_batch(pipeline: Dict[str, Any], images: List[Image.Image]) -> Dict[str, Any]:
    """
    images: lista de EXACTAMENTE 10 objetos PIL.Image (RGB), p.ej. los 10
            frames capturados por la ESP32-CAM.
    """
    if len(images) != BATCH_SIZE_REQUIRED:
        raise ValueError(
            f"Se requieren exactamente {BATCH_SIZE_REQUIRED} imágenes, "
            f"se recibieron {len(images)}."
        )

    results = [_analyze_image(pipeline, i, img) for i, img in enumerate(images)]
    top5, case = _select_top5(results, pipeline["human_conf_threshold"])

    top5_summary = [
        {
            "index": r.index,
            "human_confidence": round(r.human_confidence, 4),
            "face_confidence": round(r.face_confidence, 4),
            "has_face": r.has_face,
        }
        for r in top5
    ]

    if case == "B":
        # Caso B: sistema termina en silencio, no hay accesos que evaluar.
        return {
            "status": None,
            "matched_identity": None,
            "similarity": None,
            "top5": top5_summary,
            "silent": True,
        }

    access = _resolve_access(top5, pipeline)
    access.update({"top5": top5_summary, "silent": False})
    return access


# --------------------------------------------------------------------------
# Ejemplo de integración por parte del middleware
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import glob

    pipeline = build_pipeline("modelo_control_acceso.pth")

    image_paths = sorted(glob.glob("fotos_entrada/*.jpg"))[:BATCH_SIZE_REQUIRED]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    result = process_batch(pipeline, images)

    if result["silent"]:
        # Caso B: no se reporta nada al middleware/usuario final
        pass
    else:
        print(result["status"])
        if result["matched_identity"]:
            print("Identidad:", result["matched_identity"], "| similitud:", result["similarity"])
        for item in result["top5"]:
            print(item)
