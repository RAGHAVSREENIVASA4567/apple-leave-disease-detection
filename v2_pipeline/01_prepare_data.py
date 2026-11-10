"""
Step 1 - data preparation.

Reads data/split_manifest.csv (the fixed train/val/test split of the 419 leaf photos) and checks the
raw photos for duplicates with a perceptual hash. The dataset contains the same photo more than once,
sometimes under *different* class folders, and sometimes in both train and test. This script:
  * groups near-identical photos (hash distance <= 16 of 256 bits),
  * flags groups whose members carry different labels (label conflict -> excluded from training),
  * flags test photos that have a near-duplicate in train/val (leakage -> reported separately).

Usage:  python 01_prepare_data.py          (expects the dataset unzipped at data/raw/APPLE_DISEASE_DATASET)
Output: data/groups.csv
"""
import csv

import numpy as np
from PIL import Image

from common import HASH_DIST_DUPLICATE, ROOT, load_image


def dhash(rel_path, n=16):
    im = load_image(rel_path).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    with open(ROOT / "data" / "split_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    hashes = np.array([dhash(r["path"]) for r in rows])
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)

    # union-find over near-duplicate pairs
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ii, jj = np.where(np.triu(dist <= HASH_DIST_DUPLICATE, k=1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    roots = {}
    group = [roots.setdefault(find(i), len(roots)) for i in range(len(rows))]

    members = {}
    for i, g in enumerate(group):
        members.setdefault(g, []).append(i)
    out = []
    for i, r in enumerate(rows):
        m = members[group[i]]
        conflict = len({rows[k]["class"] for k in m}) > 1
        has_dup_elsewhere = any(rows[k]["split"] != r["split"] for k in m if k != i)
        dup_in_dev = r["split"] == "test" and any(rows[k]["split"] in ("train", "val") for k in m if k != i)
        out.append({**r, "group": group[i], "group_size": len(m), "conflict": int(conflict),
                    "cross_split_dup": int(has_dup_elsewhere), "test_leak": int(dup_in_dev)})
    with open(ROOT / "data" / "groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    n_groups = len(members)
    multi = sum(len(m) > 1 for m in members.values())
    conf_imgs = sum(o["conflict"] for o in out)
    print(f"{len(rows)} photos -> {n_groups} distinct-photo groups ({multi} groups contain duplicates)")
    print(f"photos in label-conflict groups (dropped from training): {conf_imgs}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {sum(o['split']==s for o in out)} photos, "
              f"{sum(o['split']==s and o['conflict'] for o in out)} in conflict groups")
    print(f"test photos with a near-duplicate in train/val: {sum(o['test_leak'] for o in out)}")


if __name__ == "__main__":
    main()
