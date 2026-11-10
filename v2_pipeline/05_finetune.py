"""
Fine-tuning (full transfer learning): the upper part of a pretrained MobileNetV2 is trained on the
leaf photos with strong on-the-fly augmentation, instead of keeping the CNN frozen.

Modes
  --fold K    train on cross-validation fold K's training photos, report accuracy on its held-out photos
              (same folds as 03_train_classifier.py), used to compare against the frozen-feature model
  --final     train on ALL train+val photos (label-conflict duplicates excluded) and export
              models/apple_leaf_finetuned.keras  (image 0-255 -> 4 class probabilities)

Usage:  python 05_finetune.py --fold 0 --epochs 20
        python 05_finetune.py --final --epochs 20
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras import layers

from common import CLASSES, ROOT, SEED, build_backbone, load_image, load_rows

BATCH = 16


def load_images(rows, h, w):
    cache = ROOT / "features" / f"images_{h}x{w}.npy"
    if cache.exists():
        return np.load(cache)
    arr = np.stack([np.asarray(load_image(r["path"]).resize((w, h)), dtype=np.uint8) for r in rows])
    np.save(cache, arr)
    return arr


def set_trainable(base, unfreeze_from):
    """unfreeze_from=None freezes the whole backbone; otherwise layers from that block onward are trainable
    (BatchNorm layers always stay frozen, which is standard for small-data fine-tuning)."""
    trainable = False
    for layer in base.layers:
        if unfreeze_from and layer.name.startswith(unfreeze_from):
            trainable = True
        layer.trainable = trainable and not isinstance(layer, layers.BatchNormalization)


def build_model(backbone, h, w):
    base = build_backbone(backbone)
    set_trainable(base, None)
    inp = layers.Input((h, w, 3), name="image")
    x = layers.RandomFlip("horizontal")(inp)                         # augmentation layers act only in training
    x = layers.RandomRotation(0.07, fill_mode="reflect")(x)
    x = layers.RandomZoom((-0.15, 0.1), fill_mode="reflect")(x)
    x = layers.RandomTranslation(0.05, 0.05, fill_mode="reflect")(x)
    x = layers.RandomBrightness(0.2, value_range=(0, 255))(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(CLASSES), activation="softmax", name="classifier",
                       kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(x)
    return tf.keras.Model(inp, out, name="apple_leaf_finetuned"), base


def make_ds(x, y, weights, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), tf.one_hot(y, len(CLASSES)), weights.astype("float32")))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED)
    return ds.batch(BATCH).prefetch(2)


def predict_tta(model, x):
    p = model.predict(x.astype("float32"), batch_size=16, verbose=0)
    p += model.predict(x[:, :, ::-1, :].astype("float32"), batch_size=16, verbose=0)
    return p / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mnv2_1.0")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=6)
    ap.add_argument("--unfreeze-from", default="block_9_expand")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    rows = load_rows()
    images = load_images(rows, args.height, args.width)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] in ("train", "val") and not r["conflict"]])

    if args.final:
        tr, va = dev, None
    else:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(dev, labels[dev], groups[dev]))
        tr, va = dev[folds[args.fold][0]], dev[folds[args.fold][1]]

    counts = np.bincount(labels[tr], minlength=4)
    cw = len(tr) / (4 * counts)                                  # balanced class weights
    train_ds = make_ds(images[tr], labels[tr], cw[labels[tr]], shuffle=True)

    model, base = build_model(args.backbone, args.height, args.width)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    steps_per_epoch = int(np.ceil(len(tr) / BATCH))
    ft_epochs = args.epochs - args.warmup_epochs

    def evaluate():
        if va is None:
            return ""
        acc = (model.predict(images[va].astype("float32"), batch_size=16, verbose=0).argmax(1) == labels[va]).mean()
        return f"  held-out acc {acc:.3f}"

    t0 = time.time()
    # Stage 1: train only the new classifier head (backbone frozen)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    for ep in range(args.warmup_epochs):
        h = model.fit(train_ds, epochs=1, verbose=0)
        print(f"[head ] epoch {ep + 1:2d}/{args.epochs}  loss {h.history['loss'][0]:.3f}  "
              f"train acc {h.history['accuracy'][0]:.3f}{evaluate()}  ({time.time() - t0:.0f}s)", flush=True)

    # Stage 2: unfreeze the upper backbone blocks, gentler learning rate with cosine decay
    set_trainable(base, args.unfreeze_from)
    lr = tf.keras.optimizers.schedules.CosineDecay(args.lr, ft_epochs * steps_per_epoch, alpha=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    print(f"trainable parameters now: {sum(int(np.prod(v.shape)) for v in model.trainable_weights):,} | "
          f"train photos {len(tr)}" + (f" | held-out photos {len(va)}" if va is not None else ""), flush=True)
    for ep in range(args.warmup_epochs, args.epochs):
        h = model.fit(train_ds, epochs=1, verbose=0)
        show = (ep + 1) % 3 == 0 or ep == args.epochs - 1
        print(f"[tune ] epoch {ep + 1:2d}/{args.epochs}  loss {h.history['loss'][0]:.3f}  "
              f"train acc {h.history['accuracy'][0]:.3f}{evaluate() if show else ''}  ({time.time() - t0:.0f}s)", flush=True)

    if va is not None:
        acc = (predict_tta(model, images[va]).argmax(1) == labels[va]).mean()
        print(f"\nfold {args.fold}: held-out accuracy with flip-TTA = {acc:.3f}")
    else:
        model.save(ROOT / "models" / "apple_leaf_finetuned.keras")
        with open(ROOT / "models" / "model_config_finetuned.json", "w") as f:
            json.dump({"backbone": args.backbone, "input_height": args.height, "input_width": args.width,
                       "epochs": args.epochs, "lr": args.lr, "unfreeze_from": args.unfreeze_from,
                       "classes": CLASSES, "tta": "average of image and horizontal flip",
                       "n_training_photos": int(len(tr))}, f, indent=2)
        print("saved models/apple_leaf_finetuned.keras")


if __name__ == "__main__":
    main()
