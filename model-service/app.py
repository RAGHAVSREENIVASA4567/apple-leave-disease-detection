import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

SERVICE_DIR = Path(__file__).parent
MODEL_PATH = SERVICE_DIR / "model" / "apple_disease_mobilenetv2.keras"
CLASS_NAMES = json.loads((SERVICE_DIR / "classes.json").read_text(encoding="utf-8"))
IMG_SIZE = (224, 224)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
UNCERTAIN_BELOW = 0.60
MODEL_NAME = "apple_disease_mobilenetv2"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
model(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), training=False)
print("Model ready.")


def preprocess(file_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(file_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype="float32"), axis=0)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit."), 413


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, classes=CLASS_NAMES)


@app.post("/predict")
def predict():
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify(error="Select an image before starting the analysis."), 400

    try:
        batch = preprocess(image_file.read())
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="The uploaded file is not a readable image."), 400

    probabilities = np.asarray(model(batch, training=False).numpy()[0], dtype="float64")
    if (
        probabilities.shape != (len(CLASS_NAMES),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
    ):
        app.logger.error("Model returned invalid probabilities with shape %s", probabilities.shape)
        return jsonify(error="The model returned an invalid prediction."), 500

    order = np.argsort(-probabilities)
    top_index = int(order[0])
    score = float(probabilities[top_index])
    return jsonify(
        label=CLASS_NAMES[top_index],
        title=CLASS_NAMES[top_index].title(),
        confidence=score,
        uncertain=score < UNCERTAIN_BELOW,
        probabilities=[
            {
                "label": CLASS_NAMES[int(index)],
                "title": CLASS_NAMES[int(index)].title(),
                "p": float(probabilities[index]),
            }
            for index in order
        ],
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("MODEL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODEL_PORT", "5000")),
        debug=False,
    )
