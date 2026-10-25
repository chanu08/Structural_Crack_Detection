"""
train_resnet_with_denoising.py
-------------------------------
Phase 3 — Full Pipeline:
    Image → Denoising Autoencoder (light noise) → ResNet50 → Crack / No Crack

Uses same IMG_SIZE as baseline (224) for fair comparison.
Autoencoder acts as a preprocessing/enhancement step.

Usage:
    python src/train_resnet_with_denoising.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE_AE = 64      # autoencoder size
IMG_SIZE_RN = 224     # resnet size — MUST match baseline
BATCH_SIZE  = 8       # small batch for 8GB RAM
EPOCHS_FE   = 5
EPOCHS_FT   = 5
NOISE_FACTOR= 0.1     # REDUCED noise — light enhancement
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
AE_MODEL    = "models/autoencoder.keras"
MODEL_PATH  = "models/resnet_denoised.keras"
PLOT_PATH   = "outputs/plots/resnet_denoised_history.png"
CMP_PATH    = "outputs/plots/comparison.png"
METRICS_DIR = "outputs/metrics"

# ── Load Autoencoder ──────────────────────────────────────────────────────
print("\n[1/6] Loading Denoising Autoencoder...")
if not os.path.exists(AE_MODEL):
    print(f"[ERROR] Autoencoder not found at {AE_MODEL}")
    print("   Run train_autoencoder.py first!")
    exit(1)

autoencoder = tf.keras.models.load_model(AE_MODEL)
print("   Autoencoder loaded ✅")

# ── Load Dataset ──────────────────────────────────────────────────────────
print("\n[2/6] Loading dataset...")

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE_AE, IMG_SIZE_AE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE_AE, IMG_SIZE_AE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds_raw.class_names
print(f"   Classes: {class_names}")

# ── Pipeline: Autoencoder enhancement → resize for ResNet ─────────────────
print("\n[3/6] Applying autoencoder enhancement pipeline...")

preprocess_resnet = tf.keras.applications.resnet50.preprocess_input

def enhancement_pipeline(images, labels):
    images_norm  = images / 255.0
    noise        = tf.random.normal(shape=tf.shape(images_norm), mean=0.0, stddev=NOISE_FACTOR)
    noisy        = tf.clip_by_value(images_norm + noise, 0.0, 1.0)
    enhanced     = autoencoder(noisy, training=False)
    enhanced_255 = enhanced * 255.0
    resized      = tf.image.resize(enhanced_255, (IMG_SIZE_RN, IMG_SIZE_RN))
    preprocessed = preprocess_resnet(resized)
    return preprocessed, labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.map(enhancement_pipeline, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds  = test_ds_raw.map(enhancement_pipeline,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ── Build ResNet50 ────────────────────────────────────────────────────────
print("\n[4/6] Building ResNet50 classifier...")

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE_RN, IMG_SIZE_RN, 3)
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE_RN, IMG_SIZE_RN, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs, name="ResNet50_Denoised")
model.summary()

# ── Phase 1: Feature Extraction ───────────────────────────────────────────
print(f"\n[5/6] Phase 1 — Feature Extraction ({EPOCHS_FE} epochs)...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fe = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FE,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────
print(f"\n   Phase 2 — Fine-Tuning ({EPOCHS_FT} epochs)...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FT,
    callbacks=[
        EarlyStopping(patience=2, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
)

# ── Evaluate ──────────────────────────────────────────────────────────────
print("\n[6/6] Final Evaluation...")
loss, acc = model.evaluate(test_ds)
print(f"\n   Denoised ResNet50 Test Accuracy : {acc*100:.2f}%")
print(f"   Denoised ResNet50 Test Loss     : {loss:.4f}")

os.makedirs(METRICS_DIR, exist_ok=True)
with open(f"{METRICS_DIR}/resnet_denoised_results.txt", "w") as f:
    f.write(f"Test Accuracy : {acc*100:.2f}%\n")
    f.write(f"Test Loss     : {loss:.4f}\n")

# ── Comparison Chart ──────────────────────────────────────────────────────
baseline_file = f"{METRICS_DIR}/resnet_baseline_results.txt"
if os.path.exists(baseline_file):
    with open(baseline_file) as f:
        lines = f.readlines()
    baseline_acc = float(lines[0].split(":")[1].strip().replace("%", ""))
    denoised_acc = acc * 100
    improvement  = denoised_acc - baseline_acc

    print(f"\n{'='*45}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*45}")
    print(f"  Baseline  (no denoising) : {baseline_acc:.2f}%")
    print(f"  Denoised  (with AE)      : {denoised_acc:.2f}%")
    print(f"  Improvement              : {improvement:+.2f}%")
    print(f"{'='*45}")

    os.makedirs("outputs/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Baseline ResNet50\n(No Denoising)", "ResNet50 +\nDenoising Autoencoder"],
        [baseline_acc, denoised_acc],
        color=["#e07070", "#70b870"],
        width=0.5, edgecolor="white"
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.2f}%",
            ha="center", fontsize=12, fontweight="bold"
        )
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy Comparison\nBaseline vs Denoising Autoencoder + ResNet50", fontsize=12)
    ax.set_ylim(80, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(CMP_PATH)
    print(f"\n   Comparison chart → {CMP_PATH}")
    plt.show()

# ── Training Plot ─────────────────────────────────────────────────────────
def merge(h1, h2, k):
    return h1.history[k] + h2.history[k]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(merge(history_fe, history_ft, "accuracy"),     label="Train Acc")
plt.plot(merge(history_fe, history_ft, "val_accuracy"), label="Val Acc")
plt.axvline(len(history_fe.history["accuracy"]) - 1,
            color="gray", linestyle="--", label="Fine-tune start")
plt.title("Accuracy — Denoised ResNet50")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(merge(history_fe, history_ft, "loss"),     label="Train Loss")
plt.plot(merge(history_fe, history_ft, "val_loss"), label="Val Loss")
plt.axvline(len(history_fe.history["loss"]) - 1,
            color="gray", linestyle="--", label="Fine-tune start")
plt.title("Loss — Denoised ResNet50")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"   Training plot → {PLOT_PATH}")
plt.show()

print("\n✅  Full pipeline training complete!")
print(f"   Model saved → {MODEL_PATH}")