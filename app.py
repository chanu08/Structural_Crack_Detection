"""
app.py
------
Flask web application for Structural Crack Detection.
Loads Denoising Autoencoder + ResNet50 and serves predictions via browser.

Usage:
    pip install flask
    python app.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────
IMG_SIZE_AE  = 64
IMG_SIZE_RN  = 224
NOISE_FACTOR = 0.1
AE_MODEL     = "models/autoencoder.keras"
RN_MODEL     = "models/resnet_baseline.keras"

# ── Load Models ───────────────────────────────────────────────────────────
print("Loading models...")
autoencoder = tf.keras.models.load_model(AE_MODEL)
classifier  = tf.keras.models.load_model(RN_MODEL)
print("Models loaded ✅")

preprocess_resnet = tf.keras.applications.resnet50.preprocess_input

# ── Helper: process image ─────────────────────────────────────────────────
def process_image(img_bytes):
    # Load image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    original = img.copy()

    # For autoencoder
    img_ae = img.resize((IMG_SIZE_AE, IMG_SIZE_AE))
    img_ae = np.array(img_ae) / 255.0
    img_ae = np.expand_dims(img_ae, 0).astype(np.float32)

    # Add noise
    noise  = np.random.normal(0, NOISE_FACTOR, img_ae.shape).astype(np.float32)
    noisy  = np.clip(img_ae + noise, 0, 1)

    # Denoise
    denoised = autoencoder.predict(noisy, verbose=0)

    # Resize for ResNet
    denoised_255 = denoised * 255.0
    denoised_resized = tf.image.resize(denoised_255, (IMG_SIZE_RN, IMG_SIZE_RN))
    img_rn = preprocess_resnet(denoised_resized)
    # Resize for ResNet
    denoised_255 = denoised * 255.0
    denoised_resized = tf.image.resize(denoised_255, (IMG_SIZE_RN, IMG_SIZE_RN))
    denoised_resized = np.array(denoised_resized)  # shape: (1, 224, 224, 3)
    img_rn = preprocess_resnet(denoised_resized)   # still (1, 224, 224, 3)

    # Predict
    prob  = float(classifier.predict(img_rn, verbose=0)[0][0])
    label = "Crack Detected" if prob >= 0.5 else "No Crack"
    confidence = prob if prob >= 0.5 else 1 - prob

    # Convert images to base64 for display
    def img_to_b64(arr):
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    original_resized = np.array(original.resize((IMG_SIZE_AE, IMG_SIZE_AE))) / 255.0

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "is_crack": prob >= 0.5,
        "original_img": img_to_b64(original_resized),
        "noisy_img":    img_to_b64(noisy[0]),
        "denoised_img": img_to_b64(denoised[0]),
    }

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        result = process_image(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)