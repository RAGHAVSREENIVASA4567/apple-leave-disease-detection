# Apple leaf disease detection: all project code

Classifies apple leaf photos into four classes (Apple rot, Healthy, Leaf blotch, Apple scab) with
transfer learning, and serves the model in a small web app that shows the disease name, confidence,
likely cause and suggested remedy.

## Where things are

| Folder | What it is |
|---|---|
| `v1_baseline/` | First pipeline: `preprocess.py` -> `train_model.py` (MobileNetV2 fine-tuning) -> `evaluate_model.py` |
| `v2_pipeline/` | Second pipeline (duplicate-aware, aspect-ratio-preserving, frozen features from 3 CNNs). Steps `01` to `04`; `05_finetune.py` is an experiment that did not help |
| `app/` | Web app (Flask). Uses the v1 model in `app/model/`. See `app/README.md` to run it |

## Results (held-out test set, 63 photos)

| Model | Test accuracy | 95% interval |
|---|---|---|
| v1: MobileNetV2, fine-tuned (used by the app) | 55.6% (35/63) | about 43-67% |
| v2: Xception + InceptionV3 + MobileNetV2 features, linear head | 58.7% | 46-71% |
| v2, duplicate-free subset (57 photos) | 54.4% | 40-67% |

**The 80% target was not reached.** The v2 model is not meaningfully better than v1: the difference is far
smaller than the uncertainty from having only 63 test photos. In grouped 5-fold cross-validation on
train+val, v2 scored 66.0% +/- 4.1.

## What was tried (v2)

- Keep the leaves' natural tall shape (512x256) instead of squashing them into squares (v1 did).
- Frozen ImageNet features from MobileNetV2-1.4, Xception and InceptionV3 (average + max pooled), with a
  logistic-regression head. Cross-validation: single backbones 62.5-64.5%, all three 66.0%.
- Small MLP head on the same features: 62.5-65.3% (no better).
- Fine-tuning MobileNetV2 with strong augmentation (`05_finetune.py`): 53.6% on one fold, worse than frozen features.
- Model selection used only train+val photos; the test set was evaluated once, after the choice was made.

## Problems in the dataset (they limit what any model can reach)

- **Contradictory labels:** 10 photos sit in near-duplicate groups with different labels (for example the
  same photo in both the Apple rot and Healthy folders). `01_prepare_data.py` finds them; v2 drops them from training.
- **Leakage:** 6 of the 63 test photos have a near-duplicate in train/val. `04_evaluate.py` therefore also reports
  a duplicate-free subset.
- **Background shortcut:** colour statistics from the image borders alone predict the class 46% of the time
  (always guessing the largest class gives 38%). A CNN may use background texture, so scores here may not carry over to real orchards.
- **Size:** about 350 usable photos for 4 classes.
- **"Apple rot" is not defined by the dataset.** The app's cause text assumes black rot / frogeye leaf spot; confirm it.

What would most likely help: more photos (especially healthy leaves and varied backgrounds), labels checked by a
plant pathologist, duplicates removed before splitting, and a consistent way of photographing leaves.

## Running v2 (about 30 minutes on one CPU core)

```bash
pip install -r requirements.txt
cd v2_pipeline
# unzip the dataset so that data/raw/APPLE_DISEASE_DATASET/<class folders> exists
python 01_prepare_data.py
python 02_extract_features.py --backbone mnv2_1.4     # ~5 min
python 02_extract_features.py --backbone xception     # ~12 min
python 02_extract_features.py --backbone inceptionv3  # ~9 min
python 03_train_classifier.py                         # cross-validation, then exports models/apple_leaf_classifier.keras (~190 MB)
python 04_evaluate.py                                 # test metrics and plots into reports/
```

The pretrained ImageNet weights download automatically. I tested only with local copies of the weight files
(my environment could not reach the Keras download server), so numbers on your machine may differ slightly.
If you place weight files in `v2_pipeline/weights/` the scripts use them first.

## Running v1

Run from a folder containing `raw_dataset/APPLE_DISEASE_DATASET/`:

```bash
python v1_baseline/preprocess.py
python v1_baseline/train_model.py
python v1_baseline/evaluate_model.py     # expects apple_disease_mobilenetv2.keras in the current folder
```

## The web app

`app/` serves the v1 model (small and fast). The v2 model is a ~190 MB ensemble that needs about a second per photo
on one CPU core and is not meaningfully more accurate, so it is not wired into the app.
The app is a prototype screening aid, not a diagnosis tool.
