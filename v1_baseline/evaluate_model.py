"""
Step "4. Model Evaluation" of the project flow:
    Accuracy, precision, recall, F1-score, confusion matrix

Loads the trained model produced by train_model.py and evaluates it on the
held-out TEST split (never used for training or checkpoint selection). Train and
validation metrics are also reported so over/under-fitting is visible.

Usage:
    python evaluate_model.py

Expects:
    apple_disease_mobilenetv2.keras
    processed_dataset/{train,val,test}/<class>/*.jpg

Writes to evaluation_out/:
    metrics_summary.json          overall + per-class metrics for each split
    per_class_metrics_test.csv    precision / recall / F1 / support per class (test)
    confusion_matrix_test.png     raw counts + row-normalized (recall) side by side
    per_class_metrics_test.png    grouped bar chart of precision / recall / F1
    misclassified_test.png        gallery of test images the model got wrong
"""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = Path("apple_disease_mobilenetv2.keras")
DATA_ROOT = Path("processed_dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
N_BOOTSTRAP = 2000
OUT_DIR = Path("evaluation_out")
OUT_DIR.mkdir(exist_ok=True)


def load_split(name):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / name, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,   # fixed order so predictions map back to files
    )
    return ds, ds.class_names, list(ds.file_paths)


def predict(model, ds):
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    return y_true, probs


def summarize(y_true, probs, class_names):
    y_pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[:3]
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[:3]
    return {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {
            class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(s[i])}
            for i in range(len(class_names))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist(),
    }


def bootstrap_accuracy_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    """95% bootstrap CI for accuracy. With only ~63 test images a single accuracy
    number is noisy, so the interval is the honest way to report it."""
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(float)
    accs = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(n)]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def plot_confusion(cm, class_names, path):
    cm = np.array(cm)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, data, title, fmt in [(axes[0], cm, "Confusion matrix (counts)", "d"),
                                 (axes[1], cm_norm, "Row-normalized (recall per class)", ".2f")]:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        thresh = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_per_class(per_class, path):
    names = list(per_class)
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k, (metric, label) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        vals = [per_class[n][metric] for n in names]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={per_class[n]['support']})" for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class metrics (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_misclassified(files, y_true, probs, class_names, path, max_n=12):
    y_pred = probs.argmax(axis=1)
    wrong = np.where(y_true != y_pred)[0]
    # show the most confidently wrong first -- those are the most informative errors
    wrong = wrong[np.argsort(-probs[wrong].max(axis=1))][:max_n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(plt.imread(files[idx]))
        ax.set_title(f"true: {class_names[y_true[idx]]}\npred: {class_names[y_pred[idx]]} "
                     f"({probs[idx].max():.0%})", fontsize=7)
    plt.suptitle(f"Misclassified test images (showing {len(wrong)} most confident errors)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    results = {}
    test_cache = None

    for split in ["train", "val", "test"]:
        ds, class_names, files = load_split(split)
        y_true, probs = predict(model, ds)
        results[split] = summarize(y_true, probs, class_names)
        if split == "test":
            test_cache = (files, y_true, probs, class_names)

    files, y_true, probs, class_names = test_cache
    y_pred = probs.argmax(axis=1)
    lo, hi = bootstrap_accuracy_ci(y_true, y_pred)
    results["test"]["accuracy_95ci"] = [lo, hi]

    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    t = results["test"]
    with open(OUT_DIR / "per_class_metrics_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for n, m in t["per_class"].items():
            w.writerow([n, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", m["support"]])
        w.writerow(["macro avg", f"{t['macro']['precision']:.3f}", f"{t['macro']['recall']:.3f}",
                    f"{t['macro']['f1']:.3f}", t["n_images"]])
        w.writerow(["weighted avg", f"{t['weighted']['precision']:.3f}", f"{t['weighted']['recall']:.3f}",
                    f"{t['weighted']['f1']:.3f}", t["n_images"]])

    plot_confusion(t["confusion_matrix"], class_names, OUT_DIR / "confusion_matrix_test.png")
    plot_per_class(t["per_class"], OUT_DIR / "per_class_metrics_test.png")
    plot_misclassified(files, y_true, probs, class_names, OUT_DIR / "misclassified_test.png")

    print(f"{'split':<6} {'n':>5} {'acc':>7} {'bal.acc':>8} {'macroF1':>8} {'kappa':>7}")
    for s in ["train", "val", "test"]:
        r = results[s]
        print(f"{s:<6} {r['n_images']:>5} {r['accuracy']:>7.3f} {r['balanced_accuracy']:>8.3f} "
              f"{r['macro']['f1']:>8.3f} {r['cohen_kappa']:>7.3f}")
    print(f"\nTest accuracy 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
    print("\nTest per-class:")
    print(f"{'class':<18} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>4}")
    for n, m in t["per_class"].items():
        print(f"{n:<18} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>4}")
    print("\nConfusion matrix (rows=actual, cols=predicted):", class_names)
    print(np.array(t["confusion_matrix"]))


if __name__ == "__main__":
    main()
