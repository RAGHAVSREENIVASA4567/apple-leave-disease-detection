"""
Step 2 - transfer-learning features.

Runs every leaf photo through a frozen ImageNet-pretrained CNN and stores the pooled features.
Each training/validation photo is processed several times (clean, horizontal flip, and N random
augmentations: crop / flip / rotation / brightness / contrast); test photos get clean + flip only
(flip is used for test-time augmentation). Images keep their natural tall shape (512x256) instead
of being squashed into a square.

Usage:  python 02_extract_features.py --backbone xception [--n-aug 5]
Output: features/<backbone>.npz  (feats, row_idx, variant)
"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

from common import ROOT, feature_extractor, load_rows, to_model_input, variant_image

BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["xception", "inceptionv3", "mnv2_1.4"])
    ap.add_argument("--n-aug", type=int, default=5)
    args = ap.parse_args()

    rows = load_rows()
    jobs = []
    for i, r in enumerate(rows):
        n_variants = 2 if r["split"] == "test" else 2 + args.n_aug
        jobs += [(i, v) for v in range(n_variants)]
    print(f"{args.backbone}: {len(jobs)} images to process", flush=True)

    model = feature_extractor(args.backbone)
    feats, t0 = [], time.time()
    for s in range(0, len(jobs), BATCH):
        chunk = jobs[s:s + BATCH]
        x = np.stack([to_model_input(variant_image(rows[i]["path"], v, i)) for i, v in chunk])
        feats.append(model(x, training=False).numpy())
        if (s // BATCH) % 25 == 0:
            print(f"  {s + len(chunk)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(ROOT / "features" / f"{args.backbone}.npz",
             feats=np.concatenate(feats).astype(np.float32),
             row_idx=np.array([j[0] for j in jobs]), variant=np.array([j[1] for j in jobs]))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
