"""
inference_middleware.py
------------------------
Módulo de INFERENCIA que usará el middleware.

CORRECCIÓN (origen del KeyError 'all_probabilities'):
    _match_identity() antes descartaba todas las similitudes menos la mejor,
    por lo que la clave "all_probabilities" nunca llegaba a existir en NINGUNA
    respuesta de este servicio. Ahora se calcula y se incluye siempre, en los
    3 estados de _resolve_access (AUTORIZADO / NO AUTORIZADO / SOSPECHOSO) y
    también en el Caso B (silent).

    IMPORTANTE: este cambio, por sí solo, soluciona el bug únicamente si el
    código que consume esta respuesta (ai_service.py, en el middleware)
    lee "all_probabilities" desde la respuesta cruda de Render (el campo
    "raw_response" que ai_service.py reenvía completo), y no desde el
    diccionario filtrado "best_classification" que ai_service.py arma a
    mano. Ver nota al final de este mensaje sobre esa dependencia.
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

torch.set_num_threads(1)


@dataclass
class ImageResult:
    index: int
    human_confidence: float = 0.0
    face_confidence: float = 0.0
    has_face: bool = False
    embedding: Optional[torch.Tensor] = None


def build_pipeline(pth_path: str = "modelo_control_acceso.pth") -> Dict[str, Any]:
    checkpoint = torch.load(pth_path, map_location=DEVICE)

    human_detector = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=checkpoint.get("human_detector_num_classes", 91),
    )
    human_detector.load_state_dict(checkpoint["state_dict"])
    human_detector.eval().to(DEVICE)

    mtcnn = MTCNN(
        image_size=160, margin=20, keep_all=False,
        post_process=True, device=DEVICE,
    )

    resnet = InceptionResnetV1(pretrained=None, classify=False)
    resnet.load_state_dict(checkpoint["facenet_state_dict"], strict=False)
    resnet.eval().to(DEVICE)

    return {
        "human_detector": human_detector,
        "mtcnn": mtcnn,
        "resnet": resnet,
        "authorized_faces": checkpoint["authorized_faces"],
        "human_conf_threshold": checkpoint.get("human_conf_threshold", 0.60),
        "face_match_threshold": checkpoint.get("face_match_threshold", 0.70),
    }


def _human_confidence(pipeline, image: Image.Image) -> float:
    tensor = TF.to_tensor(image).to(DEVICE)
    with torch.no_grad():
        output = pipeline["human_detector"]([tensor])[0]

    best_score = 0.0
    for label, score in zip(output["labels"], output["scores"]):
        if int(label) == COCO_PERSON_CLASS_ID:
            best_score = max(best_score, float(score))
    return best_score


def _face_detection(pipeline, image: Image.Image):
    mtcnn = pipeline["mtcnn"]

    boxes, probs = mtcnn.detect(image)
    if boxes is None or probs is None or probs[0] is None:
        return 0.0, None

    face_confidence = float(probs[0])

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


def _select_top5(results: List[ImageResult], threshold: float):
    above_threshold = [r for r in results if r.human_confidence >= threshold]

    if above_threshold:
        ranked = sorted(
            above_threshold,
            key=lambda r: (r.has_face, r.human_confidence),
            reverse=True,
        )
        return ranked[:TOP_K], "A"

    ranked = sorted(results, key=lambda r: r.human_confidence)
    return ranked[:TOP_K], "B"


def _match_identity(embedding: torch.Tensor, authorized_faces: dict, threshold: float):
    """
    Devuelve (best_name|None, best_sim, all_probabilities). all_probabilities
    es SIEMPRE un dict {nombre: score} con TODAS las personas autorizadas.
    """
    all_probabilities: Dict[str, float] = {}
    best_name, best_sim = None, -1.0

    for name, ref_embedding in authorized_faces.items():
        sim = F.cosine_similarity(
            embedding.unsqueeze(0), ref_embedding.unsqueeze(0)
        ).item()
        all_probabilities[name] = round(sim, 4)
        if sim > best_sim:
            best_name, best_sim = name, sim

    if best_name is not None and best_sim >= threshold:
        return best_name, best_sim, all_probabilities
    return None, best_sim, all_probabilities


def _resolve_access(top5: List[ImageResult], pipeline) -> Dict[str, Any]:
    faces_in_top5 = [r for r in top5 if r.has_face]

    if not faces_in_top5:
        return {
            "status": "ACCESO SOSPECHOSO",
            "matched_identity": None,
            "similarity": None,
            "all_probabilities": {},
        }

    clearest = max(faces_in_top5, key=lambda r: r.face_confidence)

    matched_name, similarity, all_probabilities = _match_identity(
        clearest.embedding,
        pipeline["authorized_faces"],
        pipeline["face_match_threshold"],
    )

    if matched_name is not None:
        return {
            "status": "ACCESO AUTORIZADO",
            "matched_identity": matched_name,
            "similarity": round(similarity, 4),
            "all_probabilities": all_probabilities,
        }

    return {
        "status": "ACCESO NO AUTORIZADO",
        "matched_identity": None,
        "similarity": round(similarity, 4) if similarity is not None else None,
        "all_probabilities": all_probabilities,
    }


def process_batch(pipeline: Dict[str, Any], images: List[Image.Image]) -> Dict[str, Any]:
    if len(images) != BATCH_SIZE_REQUIRED:
        raise ValueError(
            f"Se requieren exactamente {BATCH_SIZE_REQUIRED} imágenes, "
            f"se recibieron {len(images)}."
        )

    with torch.inference_mode():
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
        return {
            "status": None,
            "matched_identity": None,
            "similarity": None,
            "all_probabilities": {},
            "top5": top5_summary,
            "silent": True,
        }

    access = _resolve_access(top5, pipeline)
    access.update({"top5": top5_summary, "silent": False})
    return access


if __name__ == "__main__":
    import glob

    pipeline = build_pipeline("modelo_control_acceso.pth")

    image_paths = sorted(glob.glob("fotos_entrada/*.jpg"))[:BATCH_SIZE_REQUIRED]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    result = process_batch(pipeline, images)

    if result["silent"]:
        pass
    else:
        print(result["status"])
        if result["matched_identity"]:
            print("Identidad:", result["matched_identity"], "| similitud:", result["similarity"])
        print("Probabilidades:", result["all_probabilities"])
        for item in result["top5"]:
            print(item)
