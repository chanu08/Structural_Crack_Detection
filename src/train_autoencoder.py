"""
train_autoencoder.py
---------------------
Phase 2 — Convolutional Denoising Autoencoder.
Learns to remove Gaussian noise from crack images.
Saves model + sample reconstructions.

Usage:
    python src/train_autoencoder.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE    = 64
BATCH_SIZE  = 16
EPOCHS      = 20
NOISE_FACTOR= 0.3          # amount of Gaussian noise to add
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
MODEL_PATH  = "models/autoencoder.keras"
PLOT_PATH   = "outputs/plots/autoencoder_history.png"
SAMPLE_PATH = "outputs/sample_predictions/autoencoder_samples.png"

# ── Load Dataset (images only, no labels needed) ──────────────────────────
print("\n[1/5] Loading dataset...")

def load_images_only(directory, img_size, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode=None,   # no labels for autoencoder
        shuffle=True,
        seed=42
    )
    # Normalize to [0, 1]
    ds = ds.map(lambda x: x / 255.0)
    return ds

train_ds = load_images_only(TRAIN_DIR, IMG_SIZE, BATCH_SIZE)
test_ds  = load_images_only(TEST_DIR,  IMG_SIZE, BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ── Add Noise ─────────────────────────────────────────────────────────────
def add_noise(images):
    noise  = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=NOISE_FACTOR)
    noisy  = tf.clip_by_value(images + noise, 0.0, 1.0)
    return noisy

def make_noisy_pairs(clean):
    return add_noise(clean), clean   # (noisy_input, clean_target)

train_pairs = train_ds.map(make_noisy_pairs)
test_pairs  = test_ds.map(make_noisy_pairs)

# ── Build Convolutional Autoencoder ───────────────────────────────────────
print("\n[2/5] Building Denoising Autoencoder...")

def build_autoencoder(img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # ── Encoder ──
    x = layers.Conv2D(32,  (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x = layers.Conv2D(64,  (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2,2), padding="same")(x)

    # ── Decoder ──
    x = layers.Conv2DTranspose(128, (3,3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2DTranspose(64,  (3,3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2DTranspose(32,  (3,3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same")(x)

    return models.Model(inputs, decoded, name="Denoising_Autoencoder")

autoencoder = build_autoencoder(IMG_SIZE)
autoencoder.summary()

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

# ── Train ─────────────────────────────────────────────────────────────────
print("\n[3/5] Training Autoencoder...")

history = autoencoder.fit(
    train_pairs,
    validation_data=test_pairs,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
)

# ── Plot Loss ─────────────────────────────────────────────────────────────
print("\n[4/5] Saving loss plot...")
os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Autoencoder Training Loss (MSE)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"   Plot saved → {PLOT_PATH}")

# ── Visualize Reconstructions ─────────────────────────────────────────────
print("\n[5/5] Saving sample reconstructions...")
os.makedirs("outputs/sample_predictions", exist_ok=True)

sample_batch = next(iter(test_ds.take(1)))        # clean images
noisy_batch  = add_noise(sample_batch)
reconstructed= autoencoder.predict(noisy_batch)

n = 5
fig, axes = plt.subplots(3, n, figsize=(15, 6))
fig.suptitle("Top: Noisy | Middle: Reconstructed | Bottom: Original", fontsize=12)

for i in range(n):
    axes[0, i].imshow(noisy_batch[i].numpy())
    axes[0, i].axis("off")
    axes[0, i].set_title("Noisy")

    axes[1, i].imshow(np.clip(reconstructed[i], 0, 1))
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

    axes[2, i].imshow(sample_batch[i].numpy())
    axes[2, i].axis("off")
    axes[2, i].set_title("Original")

plt.tight_layout()
plt.savefig(SAMPLE_PATH)
print(f"   Samples saved → {SAMPLE_PATH}")
plt.show()

print("\n✅  Autoencoder training complete!")
print(f"   Model saved → {MODEL_PATH}")