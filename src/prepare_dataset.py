"""
prepare_dataset.py
------------------
Splits SDNET2018 raw images into train (80%) and test (20%).
Run this ONCE after extracting SDNET2018.zip into dataset/raw/

Usage:
    python src/prepare_dataset.py
"""

import os
import shutil
import random
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
RAW_DIR     = "dataset/raw"
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
SPLIT_RATIO = 0.80
SEED        = 42

EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp"}

# SDNET2018 uses these folder name patterns
CRACK_KEYS    = ["CD", "CW", "CP"]   # Cracked: Deck / Wall / Pavement
NO_CRACK_KEYS = ["UD", "UW", "UP"]   # Uncracked

random.seed(SEED)

# ── Helpers ──────────────────────────────────────────────────────────────
def collect_images(root, keys):
    images = []
    for dirpath, _, filenames in os.walk(root):
        folder = os.path.basename(dirpath)
        if any(folder.startswith(k) for k in keys):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in EXTENSIONS:
                    images.append(os.path.join(dirpath, f))
    return images


def copy_split(images, train_dest, test_dest, label):
    random.shuffle(images)
    n_train = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:n_train]
    test_imgs  = images[n_train:]

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest,  exist_ok=True)

    print(f"\n[{label}]  Total={len(images)}  Train={len(train_imgs)}  Test={len(test_imgs)}")

    for p in tqdm(train_imgs, desc=f"  train/{label}"):
        shutil.copy(p, os.path.join(train_dest, os.path.basename(p)))
    for p in tqdm(test_imgs,  desc=f"  test/{label}"):
        shutil.copy(p, os.path.join(test_dest,  os.path.basename(p)))


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SDNET2018  →  train / test split")
    print("=" * 55)

    if not os.path.exists(RAW_DIR):
        print(f"\n[ERROR] '{RAW_DIR}' not found.")
        print("  Extract SDNET2018.zip and place its contents inside dataset/raw/")
        exit(1)

    crack_imgs    = collect_images(RAW_DIR, CRACK_KEYS)
    no_crack_imgs = collect_images(RAW_DIR, NO_CRACK_KEYS)

    if not crack_imgs:
        print("[WARNING] No crack images found — check SDNET folder names.")
    if not no_crack_imgs:
        print("[WARNING] No no-crack images found — check SDNET folder names.")

    copy_split(crack_imgs,    f"{TRAIN_DIR}/crack",    f"{TEST_DIR}/crack",    "crack")
    copy_split(no_crack_imgs, f"{TRAIN_DIR}/no_crack", f"{TEST_DIR}/no_crack", "no_crack")

    print("\n✅  Done!")
    print(f"   Train  →  {TRAIN_DIR}/crack  |  {TRAIN_DIR}/no_crack")
    print(f"   Test   →  {TEST_DIR}/crack   |  {TEST_DIR}/no_crack")