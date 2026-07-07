"""
app.py
------
Servicio Flask para desplegar en Render.

Carga el pipeline (detector de humanos + facenet + authorized_faces) UNA
SOLA VEZ al arrancar el proceso (variable global a nivel de módulo). Cada
petición HTTP solo hace inferencia sobre ese pipeline ya cargado en RAM;
nunca vuelve a leer el .pth desde disco.

Endpoints:
    GET  /health   -> chequeo simple de que el servicio está vivo.
    POST /predict  -> recibe EXACTAMENTE 10 imágenes y devuelve el
                      resultado del control de acceso (Top 5 + estado).

Formatos de entrada aceptados en /predict:
    A) multipart/form-data con 10 archivos bajo el campo "images"
       (ideal si tu app Flutter sube directamente las fotos).
    B) application/json con:
           { "images": ["<base64_1>", "<base64_2>", ..., "<base64_10>"] }

Variables de entorno relevantes:
    MODEL_PATH   -> ruta al .pth (default: modelo_control_acceso.pth)
    MODEL_URL    -> si el .pth NO está en el repo (por su peso), se puede
                    alojar en un storage externo (S3, Google Drive,
                    Hugging Face Hub, etc.) y el servicio lo descarga la
                    primera vez que arranca, cacheándolo en disco.
"""

import base64
import io
import os
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image

from inference_middleware import build_pipeline, process_batch, BATCH_SIZE_REQUIRED

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_control_acceso.pth")
MODEL_URL = os.environ.get("MODEL_URL")  # opcional


def _ensure_model_downloaded():
    """Si el .pth no está presente localmente pero hay MODEL_URL, lo descarga."""
    path = Path(MODEL_PATH)
    if path.exists():
        return
    if not MODEL_URL:
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH} y no se definió MODEL_URL para descargarlo."
        )

    import urllib.request
    print(f"Descargando modelo desde MODEL_URL a {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Descarga completa.")


# --------------------------------------------------------------------
# Carga del pipeline: se ejecuta UNA sola vez, al importar este módulo
# (es decir, al arrancar el worker de gunicorn), no en cada request.
# --------------------------------------------------------------------
_ensure_model_downloaded()
print("Cargando pipeline en memoria (una sola vez)...")
PIPELINE = build_pipeline(MODEL_PATH)
print("Pipeline listo.")


def _decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": PIPELINE is not None})


@app.route("/predict", methods=["POST"])
def predict():
    images = []

    # Opción A: multipart/form-data con archivos "images"
    if request.files:
        files = request.files.getlist("images")
        for f in files:
            images.append(Image.open(f.stream).convert("RGB"))

    # Opción B: JSON con base64
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        b64_list = body.get("images", [])
        for b64_img in b64_list:
            images.append(_decode_base64_image(b64_img))

    if len(images) != BATCH_SIZE_REQUIRED:
        return jsonify({
            "error": f"Se requieren exactamente {BATCH_SIZE_REQUIRED} imágenes, "
                     f"se recibieron {len(images)}."
        }), 400

    try:
        result = process_batch(PIPELINE, images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    # Solo para pruebas locales. En Render, gunicorn es quien levanta la app.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
