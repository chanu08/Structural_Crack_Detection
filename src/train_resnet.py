"""
train_resnet.py
---------------
Phase 1 — Baseline ResNet50 classifier (no denoising).
Trains on clean crack images. Saves model + accuracy/loss plots.

Usage:
    python src/train_resnet.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN info messages

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_FE   = 10    # Feature Extraction phase
EPOCHS_FT   = 10    # Fine-Tuning phase
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
MODEL_PATH  = "models/resnet_baseline.keras"
PLOT_PATH   = "outputs/plots/resnet_baseline_history.png"

# ── Load Dataset ──────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print(f"   Classes: {class_names}")

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ── Preprocessing ─────────────────────────────────────────────────────────
preprocess = tf.keras.applications.resnet50.preprocess_input

def preprocess_ds(image, label):
    return preprocess(image), label

train_ds = train_ds.map(preprocess_ds)
test_ds  = test_ds.map(preprocess_ds)

# ── Build Model ───────────────────────────────────────────────────────────
print("\n[2/5] Building ResNet50 model...")

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # freeze for feature extraction phase

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs, name="ResNet50_Crack_Classifier")
model.summary()

# ── Phase 1: Feature Extraction ───────────────────────────────────────────
print("\n[3/5] Phase 1 — Feature Extraction (frozen base)...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fe = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FE,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────
print("\n[4/5] Phase 2 — Fine-Tuning (last 30 layers unfrozen)...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # smaller LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FT,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
)

# ── Evaluate ──────────────────────────────────────────────────────────────
print("\n[5/5] Final Evaluation...")
loss, acc = model.evaluate(test_ds)
print(f"\n   Baseline ResNet50 Test Accuracy : {acc*100:.2f}%")
print(f"   Baseline ResNet50 Test Loss     : {loss:.4f}")

# Save accuracy to file
os.makedirs("outputs/metrics", exist_ok=True)
with open("outputs/metrics/resnet_baseline_results.txt", "w") as f:
    f.write(f"Test Accuracy : {acc*100:.2f}%\n")
    f.write(f"Test Loss     : {loss:.4f}\n")

# ── Plot History ──────────────────────────────────────────────────────────
def merge_history(h1, h2, key):
    return h1.history[key] + h2.history[key]

acc_all     = merge_history(history_fe, history_ft, "accuracy")
val_acc_all = merge_history(history_fe, history_ft, "val_accuracy")
loss_all    = merge_history(history_fe, history_ft, "loss")
val_loss_all= merge_history(history_fe, history_ft, "val_loss")

os.makedirs("outputs/plots", exist_ok=True)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc_all,     label="Train Acc")
plt.plot(val_acc_all, label="Val Acc")
plt.axvline(EPOCHS_FE - 1, color="gray", linestyle="--", label="Fine-tune start")
plt.title("Accuracy — ResNet50 Baseline")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_all,     label="Train Loss")
plt.plot(val_loss_all, label="Val Loss")
plt.axvline(EPOCHS_FE - 1, color="gray", linestyle="--", label="Fine-tune start")
plt.title("Loss — ResNet50 Baseline")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\n   Plot saved → {PLOT_PATH}")
plt.show()