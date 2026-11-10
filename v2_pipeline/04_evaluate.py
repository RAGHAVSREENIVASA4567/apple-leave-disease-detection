"""
Step 4 - model evaluation on the held-out TEST photos.

Runs the exported Keras model end-to-end on the raw test photos (clean + horizontal-flip, probabilities
averaged) and reports accuracy, precision, recall, F1 and the confusion matrix.

Because the dataset contains duplicate photos, results are given twice:
  * all 63 test photos, and
  * the "duplicate-free" subset: test photos with no near-duplicate in train/val and no label conflict.
The 95% interval is a bootstrap over test photos - with so few photos it is wide, and it should be
quoted next to the headline accuracy.

Usage:  python 04_evaluate.py
Output: reports/test_metrics.json, per_class_metrics.csv, predictions.csv,
        confusion_matrix.png, per_class_metrics.png, misclassified.png
"""
import csv
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from common import CLASSES, ROOT, SEED, load_image, load_rows, to_model_input

BATCH = 8


def predict_tta(model, rows):
    probs = []
    for s in range(0, len(rows), BATCH):
        imgs = [to_model_input(load_image(r["path"])) for r in rows[s:s + BATCH]]
        x = np.stack(imgs)
        p = model(x, training=False).numpy() + model(x[:, :, ::-1, :], training=False).numpy()
        probs.append(p / 2)
    return np.concatenate(probs)


def bootstrap_ci(correct, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    accs = [rng.choice(correct, len(correct)).mean() for _ in range(n)]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    ci = bootstrap_ci((y == pred).astype(float))
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)), "accuracy_95ci": ci,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(CLASSES)},
            "confusion_matrix": confusion_matrix(y, pred, labels=range(4)).tolist()}


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, d, t, fmt in [(axes[0], cm, "counts", "d"), (axes[1], norm, "row-normalized (recall)", ".2f")]:
        im = ax.imshow(d, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, format(d[i, j], fmt), ha="center", va="center", color="white" if d[i, j] > d.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{title}: {t}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_per_class(pc, path):
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k, (m, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        v = [pc[c][m] for c in CLASSES]
        bars = ax.bar(x + (k - 1) * w, v, w, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={pc[c]['support']})" for c in CLASSES], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title("Per-class metrics (test set)"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_misclassified(rows, y, probs, path, max_n=12):
    pred = probs.argmax(1)
    wrong = np.where(y != pred)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:max_n]
    if len(wrong) == 0:
        return
    cols = 6; nrows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 4.2 * nrows)); axes = np.atleast_2d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, wrong):
        a.imshow(load_image(rows[i]["path"]))
        a.set_title(f"true: {CLASSES[y[i]][:10]}\npred: {CLASSES[pred[i]][:10]} ({probs[i].max():.0%})", fontsize=7)
    plt.suptitle(f"Misclassified test photos ({len(wrong)} most confident errors shown)", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    import tensorflow as tf
    model = tf.keras.models.load_model(ROOT / "models" / "apple_leaf_classifier.keras")
    rows = [r for r in load_rows() if r["split"] == "test"]
    y = np.array([r["label"] for r in rows])
    probs = predict_tta(model, rows)
    pred = probs.argmax(1)

    clean = np.array([not (r["conflict"] or int(r["test_leak"])) for r in rows])
    out = {"all_test": metrics(y, pred), "duplicate_free_test": metrics(y[clean], pred[clean])}
    with open(ROOT / "reports" / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(ROOT / "reports" / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred", "confidence", "in_duplicate_free_subset"] + CLASSES)
        for r, t, p, pr, c in zip(rows, y, pred, probs, clean):
            w.writerow([r["path"], CLASSES[t], CLASSES[p], f"{pr.max():.3f}", int(c)] + [f"{v:.3f}" for v in pr])
    with open(ROOT / "reports" / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "precision", "recall", "f1", "support"])
        for c, m in out["all_test"]["per_class"].items():
            w.writerow([c, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", m["support"]])
        m = out["all_test"]["macro"]
        w.writerow(["macro avg", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", out["all_test"]["n"]])

    plot_confusion(out["all_test"]["confusion_matrix"], ROOT / "reports" / "confusion_matrix.png", "Test set")
    plot_per_class(out["all_test"]["per_class"], ROOT / "reports" / "per_class_metrics.png")
    plot_misclassified(rows, y, probs, ROOT / "reports" / "misclassified.png")

    for name, m in out.items():
        lo, hi = m["accuracy_95ci"]
        print(f"\n[{name}] n={m['n']}  accuracy {m['accuracy']:.3f} (95% CI {lo:.3f}-{hi:.3f})  "
              f"balanced acc {m['balanced_accuracy']:.3f}  macro P/R/F1 "
              f"{m['macro']['precision']:.3f}/{m['macro']['recall']:.3f}/{m['macro']['f1']:.3f}")
    print("\nper-class (all test):")
    for c, m in out["all_test"]["per_class"].items():
        print(f"  {c:18s} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")
    print("confusion matrix (rows=actual):")
    print(np.array(out["all_test"]["confusion_matrix"]))


if __name__ == "__main__":
    main()
