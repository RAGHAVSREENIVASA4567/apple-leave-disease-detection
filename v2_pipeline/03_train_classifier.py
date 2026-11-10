"""
Step 3 - model training (transfer learning).

Frozen ImageNet CNN features (from 02_extract_features.py) + a trained softmax classifier head.

Model selection uses ONLY the train+val photos (never the test set):
  * photos in label-conflict duplicate groups are excluded,
  * 5-fold cross-validation that keeps near-duplicate photos in the same fold (StratifiedGroupKFold),
  * training uses every augmented variant of the training photos; validation uses clean + flipped
    (test-time augmentation: probabilities averaged).
The best (backbones, C) combination is then refit on all train+val photos and exported as a single
Keras model (CNN backbones + classifier head) that takes a 512x256 RGB image (0-255) and returns the
4 class probabilities.

Usage:  python 03_train_classifier.py [--no-export]
Output: reports/cv_results.csv, models/apple_leaf_classifier.keras, models/model_config.json
"""
import argparse
import csv
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, IMG_H, IMG_W, ROOT, SEED, load_rows

BACKBONE_SETS = [("mnv2_1.4",), ("xception",), ("inceptionv3",), ("xception", "inceptionv3"),
                 ("xception", "inceptionv3", "mnv2_1.4")]
C_VALUES = [0.002, 0.01, 0.05, 0.25]


def load_features(backbones):
    """Concatenate features of several backbones (extraction order is identical, so rows align)."""
    parts, ref = [], None
    for b in backbones:
        z = np.load(ROOT / "features" / f"{b}.npz")
        key = (z["row_idx"], z["variant"])
        if ref is None:
            ref = key
        assert (ref[0] == key[0]).all() and (ref[1] == key[1]).all(), "feature files are not aligned"
        parts.append(z["feats"])
    return np.concatenate(parts, axis=1), ref[0], ref[1]


def index_by_row(row_idx, variant):
    table = {}
    for k, (i, v) in enumerate(zip(row_idx, variant)):
        table.setdefault(int(i), {})[int(v)] = k
    return table


def make_clf(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=300, tol=1e-3, class_weight="balanced"))


def train_matrix(rows_idx, table, feats, labels):
    ks = [k for i in rows_idx for k in table[i].values()]
    y = np.concatenate([[labels[i]] * len(table[i]) for i in rows_idx])
    return feats[ks], y


def predict_tta(clf, rows_idx, table, feats):
    """Average class probabilities of the clean (variant 0) and horizontally flipped (variant 1) photo."""
    p0 = clf.predict_proba(feats[[table[i][0] for i in rows_idx]])
    p1 = clf.predict_proba(feats[[table[i][1] for i in rows_idx]])
    return (p0 + p1) / 2


def export_keras_model(backbones, clf, out_path):
    """Backbones + head in one Keras model. The StandardScaler is folded into the Dense layer, so the
    model works directly on raw pooled features and needs no sklearn at inference time."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from common import feature_extractor

    scaler, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    scale = scaler.scale_
    W = (lr.coef_ / scale).T.astype("float32")                    # (D, 4)
    b = (lr.intercept_ - (lr.coef_ * scaler.mean_ / scale).sum(1)).astype("float32")

    inp = layers.Input((IMG_H, IMG_W, 3), name="image")
    feats = [feature_extractor(name)(inp) for name in backbones]
    x = feats[0] if len(feats) == 1 else layers.Concatenate()(feats)
    out = layers.Dense(len(CLASSES), activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inp, out, name="apple_leaf_classifier")
    model.get_layer("classifier").set_weights([W, b])
    model.save(out_path)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] in ("train", "val") and not r["conflict"]])
    print(f"model selection on {len(dev)} train+val photos "
          f"({sum(r['conflict'] for r in rows if r['split'] != 'test')} dropped for label conflicts)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(dev, labels[dev], groups[dev]))
    results = []
    for backbones in BACKBONE_SETS:
        if not all((ROOT / "features" / f"{b}.npz").exists() for b in backbones):
            continue
        feats, ridx, var = load_features(backbones)
        table = index_by_row(ridx, var)
        for C in C_VALUES:
            t0, accs, f1s = time.time(), [], []
            for tr, va in folds:
                Xtr, ytr = train_matrix(dev[tr], table, feats, labels)
                clf = make_clf(C).fit(Xtr, ytr)
                pred = predict_tta(clf, dev[va], table, feats).argmax(1)
                accs.append(accuracy_score(labels[dev[va]], pred))
                f1s.append(f1_score(labels[dev[va]], pred, average="macro"))
            res = dict(backbones="+".join(backbones), C=C, cv_acc=np.mean(accs), cv_acc_std=np.std(accs),
                       cv_macro_f1=np.mean(f1s))
            results.append(res)
            print(f"{res['backbones']:28s} C={C:<7} CV acc {res['cv_acc']:.3f} +/- {res['cv_acc_std']:.3f}  "
                  f"macro-F1 {res['cv_macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    with open(ROOT / "reports" / "cv_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    best = max(results, key=lambda r: (r["cv_acc"], r["cv_macro_f1"]))
    print(f"\nbest by cross-validation: {best['backbones']}  C={best['C']}  "
          f"CV accuracy {best['cv_acc']:.3f}")

    if args.no_export:
        return
    backbones = tuple(best["backbones"].split("+"))
    feats, ridx, var = load_features(backbones)
    table = index_by_row(ridx, var)
    Xall, yall = train_matrix(dev, table, feats, labels)
    clf = make_clf(best["C"]).fit(Xall, yall)
    model = export_keras_model(backbones, clf, ROOT / "models" / "apple_leaf_classifier.keras")
    with open(ROOT / "models" / "model_config.json", "w") as f:
        json.dump({"backbones": list(backbones), "C": best["C"], "classes": CLASSES,
                   "input_height": IMG_H, "input_width": IMG_W, "tta": "average of image and horizontal flip",
                   "cv_accuracy": best["cv_acc"], "cv_accuracy_std": best["cv_acc_std"],
                   "n_training_photos": int(len(dev))}, f, indent=2)
    print("exported models/apple_leaf_classifier.keras")


if __name__ == "__main__":
    main()
