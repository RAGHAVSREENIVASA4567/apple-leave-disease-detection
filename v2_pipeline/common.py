"""Shared helpers for the apple leaf disease pipeline (data loading, augmentation, backbones)."""
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", ROOT / "weights"))
CLASSES = ["APPLE ROT LEAVES", "HEALTHY LEAVES", "LEAF BLOTCH", "SCAB LEAVES"]
IMG_H, IMG_W = 512, 256          # leaves are ~1:2 tall photos; keep that shape instead of squashing to a square
SEED = 42
HASH_DIST_DUPLICATE = 16         # 256-bit dHash distance at/below which two photos are treated as the same photo


# ------------------------------- data -----------------------------------
def load_rows():
    """Rows of data/groups.csv (made by 01_prepare_data.py)."""
    with open(ROOT / "data" / "groups.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = CLASSES.index(r["class"])
        r["group"] = int(r["group"])
        r["conflict"] = r["conflict"] == "1"
    return rows


def load_image(rel_path):
    img = Image.open(RAW_DIR / rel_path)
    return ImageOps.exif_transpose(img).convert("RGB")


def to_model_input(img):
    """PIL image -> float32 array (H, W, 3) in 0-255 at the model's input size."""
    return np.asarray(img.resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.float32)


def augment(img, rng):
    """Random crop, flips, rotation, brightness/contrast. Colour hue is left alone because lesion colour matters."""
    w, h = img.size
    s = rng.uniform(0.80, 1.0)
    cw, ch = int(w * s), min(h, int(h * s * rng.uniform(0.95, 1.05)))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.3:
        img = ImageOps.flip(img)
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    img = img.rotate(float(rng.uniform(-25, 25)), resample=Image.BILINEAR, fillcolor=fill)
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.75, 1.25)))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
    return img


def variant_image(rel_path, variant, row_idx):
    """variant 0 = clean, 1 = horizontal flip, >=2 = seeded random augmentation (reproducible)."""
    img = load_image(rel_path)
    if variant == 0:
        return img
    if variant == 1:
        return ImageOps.mirror(img)
    rng = np.random.default_rng([SEED, row_idx, variant])
    return augment(img, rng)


# ----------------------------- backbones --------------------------------
def _backbones():
    from tensorflow.keras import applications as A
    return {
        "xception": (A.Xception, "xception_notop.h5"),
        "inceptionv3": (A.InceptionV3, "inceptionv3_notop.h5"),
        "mnv2_1.4": (lambda **k: A.MobileNetV2(alpha=1.4, **k), "mobilenetv2_1.4_224_notop.h5"),
        "mnv2_1.0": (lambda **k: A.MobileNetV2(alpha=1.0, **k), "mobilenetv2_1.0_224_notop.h5"),
    }


def build_backbone(name):
    """ImageNet-pretrained convolutional base (no classifier). Uses a local weight file in
    WEIGHTS_DIR if present (for machines that can't reach the Keras download server), otherwise
    downloads the standard ImageNet weights."""
    ctor, local = _backbones()[name]
    shape = (None, None, 3)
    local_path = WEIGHTS_DIR / local
    if local_path.exists():
        model = ctor(weights=None, include_top=False, input_shape=shape)
        model.load_weights(str(local_path))
    else:
        model = ctor(weights="imagenet", include_top=False, input_shape=shape)
    model.trainable = False
    return model


def feature_extractor(name):
    """Image (0-255) -> [global-average-pool, global-max-pool] of the last conv map.
    Max pooling keeps 'is there a small lesion anywhere' information that averaging dilutes."""
    import tensorflow as tf
    from tensorflow.keras import layers
    base = build_backbone(name)
    inp = layers.Input((IMG_H, IMG_W, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)       # [-1, 1] input range used by these backbones
    x = base(x, training=False)
    out = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    return tf.keras.Model(inp, out, name=f"features_{name}")
