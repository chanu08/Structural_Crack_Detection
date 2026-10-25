"""
predict.py
----------
Run crack detection on a single image.
Uses the full pipeline: Autoencoder → ResNet50.

Usage:
    python src/predict.py --image path/to/image.jpg
"""

import os, argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE_AE = 128
IMG_SIZE_RN = 224
NOISE_FACTOR= 0.3
AE_MODEL    = "models/autoencoder.keras"
RN_MODEL    = "models/resnet_denoised.keras"

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

img_path = args.image
if not os.path.exists(img_path):
    print(f"[ERROR] Image not found: {img_path}")
    exit(1)

# ── Load Models ───────────────────────────────────────────────────────────
print("Loading models...")
autoencoder = tf.keras.models.load_model(AE_MODEL)
classifier  = tf.keras.models.load_model(RN_MODEL)

# ── Load & Preprocess Image ───────────────────────────────────────────────
img_orig = cv2.imread(img_path)
img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

# For autoencoder
img_ae = cv2.resize(img_orig, (IMG_SIZE_AE, IMG_SIZE_AE)) / 255.0
img_ae = np.expand_dims(img_ae, 0).astype(np.float32)

# Add noise
noise   = np.random.normal(0, NOISE_FACTOR, img_ae.shape).astype(np.float32)
noisy   = np.clip(img_ae + noise, 0, 1)

# Denoise
denoised = autoencoder.predict(noisy)

# Resize for ResNet
denoised_255 = denoised * 255.0
img_rn = tf.image.resize(denoised_255, (IMG_SIZE_RN, IMG_SIZE_RN))
img_rn = tf.keras.applications.resnet50.preprocess_input(img_rn)
img_rn = np.expand_dims(img_rn, 0)

# ── Predict ───────────────────────────────────────────────────────────────
prob = classifier.predict(img_rn)[0][0]
label = "CRACK DETECTED" if prob >= 0.5 else "NO CRACK"
confidence = prob if prob >= 0.5 else 1 - prob

print(f"\n{'='*40}")
print(f"  Result     : {label}")
print(f"  Confidence : {confidence*100:.1f}%")
print(f"{'='*40}\n")

# ── Show Result ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"Prediction: {label}  ({confidence*100:.1f}%)", fontsize=13,
             color="red" if label == "CRACK DETECTED" else "green")

axes[0].imshow(img_orig)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(np.clip(noisy[0], 0, 1))
axes[1].set_title("Noisy Input")
axes[1].axis("off")

axes[2].imshow(np.clip(denoised[0], 0, 1))
axes[2].set_title("Denoised")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_predictions/last_prediction.png")
plt.show()