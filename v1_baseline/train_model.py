"""
Step "3. Model Training" of the project flow: CNN via transfer learning (MobileNetV2).

Reads the already-preprocessed dataset from processed_dataset/{train,val,test}/<class>/*.jpg
(produced by preprocess.py -- resized to 224x224, augmented on the training split).

Pipeline:
    1. Load train/val/test splits with tf.keras.utils.image_dataset_from_directory
    2. Build MobileNetV2 (ImageNet-pretrained, frozen) + a small classifier head
    3. Phase 1: train only the new head (base frozen)
    4. Phase 2: unfreeze the top of the base and fine-tune at a low learning rate
    5. Evaluate on the held-out test set (accuracy, per-class report, confusion matrix)
    6. Save the trained model, training curves, and confusion matrix

Note on pretrained weights: tf.keras.applications' default weights download from
storage.googleapis.com, which isn't reachable on this network, so the ImageNet
weights are loaded from a local file (see MOBILENETV2_WEIGHTS below) sourced from
the original MobileNetV2-for-Keras release, which uses the same architecture/layer
names as tf.keras.applications.MobileNetV2 -- verified to load cleanly.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Config -----------------------------------
DATA_ROOT = Path("processed_dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
MOBILENETV2_WEIGHTS = "/tmp/mobilenetv2_notop.h5"   # local ImageNet weights (no top)
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 6
FINE_TUNE_AT_LAYER = 120     # unfreeze from this layer index onward (of 154 total)
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5

OUT_DIR = Path("model_out")
OUT_DIR.mkdir(exist_ok=True)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=True, seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (train_ds.prefetch(autotune), val_ds.prefetch(autotune),
            test_ds.prefetch(autotune), class_names)


def build_model(num_classes):
    import os
    if os.path.exists(MOBILENETV2_WEIGHTS):          # local weight file (used where the Keras download is blocked)
        base = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
        base.load_weights(MOBILENETV2_WEIGHTS)
    else:                                            # normal case: standard ImageNet weights
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)          # matches step-2 normalization
    x = layers.Rescaling(2.0, offset=-1.0)(x)         # -> [-1,1], what MobileNetV2 expects
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model, base


def class_weights_from(train_ds, num_classes):
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return dict(enumerate(weights))


def plot_history(hist1, hist2, path):
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]
    switch = len(hist1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(switch - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(switch - 0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print("Classes:", class_names)

    model, base = build_model(num_classes)
    cw = class_weights_from(train_ds, num_classes)
    print("Class weights (balanced):", {class_names[k]: round(v, 2) for k, v in cw.items()})

    # A fresh EarlyStopping per phase controls *when that phase stops* (so phase 1's
    # best score doesn't prematurely cut phase 2 short). A single ModelCheckpoint
    # shared across both phases tracks the *true best model overall* -- whichever
    # epoch, in either phase, had the lowest val_loss -- so fine-tuning can never
    # silently leave us worse off than phase 1 alone.
    ckpt_path = OUT_DIR / "best_checkpoint.keras"
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True)

    # --- Phase 1: train the head only, base frozen ---
    model.compile(optimizer=tf.keras.optimizers.Adam(HEAD_LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Phase 1: training classifier head (base frozen) ===")
    es1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
                       class_weight=cw, callbacks=[es1, ckpt], verbose=2)

    # --- Phase 2: fine-tune the top of the base at a low learning rate ---
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"\n=== Phase 2: fine-tuning from layer {FINE_TUNE_AT_LAYER} onward ===")
    es2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                       class_weight=cw, callbacks=[es2, ckpt], verbose=2)

    # Load the true best-across-both-phases model before evaluating/saving
    model = tf.keras.models.load_model(ckpt_path)
    print(f"\nLoaded overall best checkpoint (lowest val_loss across both phases) from {ckpt_path}")

    # --- Evaluate on test set ---
    print("\n=== Test set evaluation ===")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.3f} | Test loss: {test_loss:.3f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    cm = confusion_matrix(y_true, y_pred)

    # --- Save everything ---
    model.save(OUT_DIR / "apple_disease_mobilenetv2.keras")
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n{report}")
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump({"phase1": hist1.history, "phase2": hist2.history,
                   "test_accuracy": float(test_acc), "test_loss": float(test_loss),
                   "class_names": class_names}, f, indent=2)

    plot_history(hist1, hist2, OUT_DIR / "training_curves.png")
    plot_confusion(cm, class_names, OUT_DIR / "confusion_matrix.png")
    print(f"\nSaved model + report + plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
