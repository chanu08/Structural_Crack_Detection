"""
evaluate.py
-----------
Evaluates the trained denoised ResNet50 model.
Outputs: accuracy, confusion matrix, classification report.

Usage:
    python src/evaluate.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE_AE = 128
IMG_SIZE_RN = 224
BATCH_SIZE  = 32
NOISE_FACTOR= 0.3
TEST_DIR    = "dataset/test"
AE_MODEL    = "models/autoencoder.keras"
RN_MODEL    = "models/resnet_denoised.keras"
OUT_DIR     = "outputs/metrics"

# ── Load Models ───────────────────────────────────────────────────────────
print("\n[1/4] Loading models...")
autoencoder = tf.keras.models.load_model(AE_MODEL)
classifier  = tf.keras.models.load_model(RN_MODEL)

# ── Load Test Dataset ─────────────────────────────────────────────────────
print("\n[2/4] Loading test dataset...")
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE_AE, IMG_SIZE_AE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds_raw.class_names
print(f"   Classes: {class_names}")

preprocess_resnet = tf.keras.applications.resnet50.preprocess_input

def pipeline(images, labels):
    images_norm = images / 255.0
    noise   = tf.random.normal(shape=tf.shape(images_norm), mean=0.0, stddev=NOISE_FACTOR)
    noisy   = tf.clip_by_value(images_norm + noise, 0.0, 1.0)
    denoised= autoencoder(noisy, training=False)
    resized = tf.image.resize(denoised * 255.0, (IMG_SIZE_RN, IMG_SIZE_RN))
    return preprocess_resnet(resized), labels

AUTOTUNE = tf.data.AUTOTUNE
test_ds  = test_ds_raw.map(pipeline, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ── Get Predictions ───────────────────────────────────────────────────────
print("\n[3/4] Running predictions...")
y_true, y_pred = [], []

for images, labels in test_ds:
    preds = classifier.predict(images, verbose=0)
    y_true.extend(labels.numpy().flatten())
    y_pred.extend((preds.flatten() >= 0.5).astype(int))

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

# ── Print Report ──────────────────────────────────────────────────────────
print("\n[4/4] Results")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=class_names))

acc = np.mean(y_true == y_pred)
print(f"  Overall Accuracy: {acc*100:.2f}%")
print("=" * 50)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/evaluation_report.txt", "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=class_names))
    f.write(f"\nOverall Accuracy: {acc*100:.2f}%\n")

# ── Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix  (Accuracy: {acc*100:.2f}%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png")
print(f"\n   Confusion matrix saved → {OUT_DIR}/confusion_matrix.png")
plt.show()

print("\n✅  Evaluation complete!")