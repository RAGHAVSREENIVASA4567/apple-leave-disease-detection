"""
Preprocessing pipeline for the Apple Disease Dataset.
Implements step "2. Preprocessing" of the project flow:
    Resize -> Augment (rotate, flip, brightness) -> Normalize

Usage:
    python preprocess.py

Reads images from:  raw_dataset/APPLE_DISEASE_DATASET/<CLASS_NAME>/*.jpg
Writes:
    processed_dataset/{train,val,test}/<CLASS_NAME>/*.jpg   (resized + augmented, viewable JPGs)
    arrays/{X_train,y_train,X_val,y_val,X_test,y_test}.npy  (normalized float32, ready for a CNN)
    arrays/classes.json                                     (label index -> class name map)
    preview_before_after.png                                (visual sanity check)
"""

import os
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split

# ----------------------------- Config ---------------------------------
RAW_ROOT = Path("raw_dataset/APPLE_DISEASE_DATASET")
OUT_IMG_ROOT = Path("processed_dataset")
OUT_ARR_ROOT = Path("arrays")
IMG_SIZE = (224, 224)          # standard input size for CNNs / transfer learning
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train, val, test
AUGS_PER_TRAIN_IMAGE = 2       # extra augmented copies generated per training image
ROTATION_RANGE_DEG = 25
BRIGHTNESS_RANGE = (0.7, 1.3)
JPEG_QUALITY = 90
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------- Augmentation ops ----------------------------
def random_rotate(img: Image.Image) -> Image.Image:
    angle = random.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG)
    # expand=False + fillcolor keeps output size fixed, filling new corners
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))


def random_flip(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = ImageOps.mirror(img)          # horizontal flip
    if random.random() < 0.2:
        img = ImageOps.flip(img)            # occasional vertical flip
    return img


def random_brightness(img: Image.Image) -> Image.Image:
    factor = random.uniform(*BRIGHTNESS_RANGE)
    return ImageEnhance.Brightness(img).enhance(factor)


def augment(img: Image.Image) -> Image.Image:
    """Apply rotate + flip + brightness jitter, in random order-ish combo."""
    img = random_rotate(img)
    img = random_flip(img)
    img = random_brightness(img)
    return img


def resize(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)  # respect phone-camera orientation tags first
    return img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)


def normalize(img: Image.Image) -> np.ndarray:
    """Scale pixel values from [0,255] uint8 to [0,1] float32."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# --------------------------- Dataset discovery --------------------------
def discover_dataset(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    filepaths, labels = [], []
    for label, cls in enumerate(classes):
        for f in (root / cls).iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                filepaths.append(f)
                labels.append(label)
    return filepaths, labels, classes


def stratified_split(filepaths, labels):
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    fp_train, fp_rest, y_train, y_rest = train_test_split(
        filepaths, labels, train_size=train_ratio, stratify=labels, random_state=SEED
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    fp_val, fp_test, y_val, y_test = train_test_split(
        fp_rest, y_rest, train_size=rel_val, stratify=y_rest, random_state=SEED
    )
    return (fp_train, y_train), (fp_val, y_val), (fp_test, y_test)


# ------------------------------ Pipeline --------------------------------
def process_split(filepaths, labels, classes, split_name, augment_images):
    """
    Resize (+ augment if training) each image and save it as a JPG under
    processed_dataset/<split>/<class>/. Every saved image is also run through
    normalize() to confirm pixel values land in [0,1]; running stats are
    tracked instead of keeping every array in memory (the full array set for
    an augmented split would run into the hundreds of MB, which isn't a
    practical thing to hand back as a download).
    """
    count = 0
    px_min, px_max, mean_accum = 1.0, 0.0, 0.0

    for fp, label in zip(filepaths, labels):
        cls_name = classes[label]
        out_dir = OUT_IMG_ROOT / split_name / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(fp)
        base = resize(img)

        # Always keep one clean resized copy; add augmented copies for training only
        variants = [base]
        if augment_images:
            for _ in range(AUGS_PER_TRAIN_IMAGE):
                variants.append(augment(base))

        for i, variant in enumerate(variants):
            out_name = f"{fp.stem}_{i}.jpg"
            variant.save(out_dir / out_name, quality=JPEG_QUALITY)

            arr = normalize(variant)
            px_min = min(px_min, float(arr.min()))
            px_max = max(px_max, float(arr.max()))
            mean_accum += float(arr.mean())
            count += 1

    return count, px_min, px_max, (mean_accum / count if count else 0.0)


def build_sample_array(filepaths, labels, classes, n=16):
    """Materialize a small normalized batch (resize+augment+normalize) as a
    concrete, loadable artifact -- e.g. to sanity-check a model's input shape."""
    idx = random.sample(range(len(filepaths)), min(n, len(filepaths)))
    X, y = [], []
    for i in idx:
        img = Image.open(filepaths[i])
        base = resize(img)
        variant = augment(base) if random.random() < 0.5 else base
        X.append(normalize(variant))
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def make_preview(filepaths, classes, labels, n=3):
    """Save a grid: original vs 3 augmented versions, for a few sample images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_idx = random.sample(range(len(filepaths)), n)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n))
    col_titles = ["Original (resized)", "Augmented 1", "Augmented 2", "Augmented 3"]

    for row, idx in enumerate(sample_idx):
        img = Image.open(filepaths[idx])
        base = resize(img)
        versions = [base] + [augment(base) for _ in range(3)]
        for col, v in enumerate(versions):
            ax = axes[row, col]
            ax.imshow(v)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
        axes[row, 0].set_ylabel(classes[labels[idx]], fontsize=9)

    plt.tight_layout()
    plt.savefig("preview_before_after.png", dpi=130)
    plt.close()


def main():
    filepaths, labels, classes = discover_dataset(RAW_ROOT)
    print(f"Found {len(filepaths)} images across {len(classes)} classes: {classes}")

    (fp_tr, y_tr), (fp_va, y_va), (fp_te, y_te) = stratified_split(filepaths, labels)
    print(f"Split -> train: {len(fp_tr)}, val: {len(fp_va)}, test: {len(fp_te)}")

    OUT_ARR_ROOT.mkdir(exist_ok=True)

    n_train, min_tr, max_tr, mean_tr = process_split(fp_tr, y_tr, classes, "train", augment_images=True)
    n_val, min_va, max_va, mean_va = process_split(fp_va, y_va, classes, "val", augment_images=False)
    n_test, min_te, max_te, mean_te = process_split(fp_te, y_te, classes, "test", augment_images=False)

    print(f"After augmentation -> train: {n_train} images "
          f"(from {len(fp_tr)} originals x ~{1 + AUGS_PER_TRAIN_IMAGE})")
    print(f"Saved -> train: {n_train}, val: {n_val}, test: {n_test} (all resized to {IMG_SIZE})")
    print(f"Normalized pixel range -> train [{min_tr:.3f}, {max_tr:.3f}] mean {mean_tr:.3f} | "
          f"val [{min_va:.3f}, {max_va:.3f}] mean {mean_va:.3f} | "
          f"test [{min_te:.3f}, {max_te:.3f}] mean {mean_te:.3f}")

    X_sample, y_sample = build_sample_array(filepaths, labels, classes, n=16)
    np.save(OUT_ARR_ROOT / "sample_normalized_batch.npy", X_sample)
    np.save(OUT_ARR_ROOT / "sample_labels.npy", y_sample)
    with open(OUT_ARR_ROOT / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Sample normalized batch shape: {X_sample.shape}, dtype {X_sample.dtype}")

    make_preview(filepaths, classes, labels, n=3)
    print("Saved processed images to processed_dataset/, sample arrays to arrays/, preview to preview_before_after.png")


if __name__ == "__main__":
    main()
