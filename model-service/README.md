# MobileNetV2 model service

This Flask service loads `model/apple_disease_mobilenetv2.keras` and provides
the four-class inference endpoint used by the AppleCare AI Node backend. The
React app's existing upload, results, dashboard, and history flows are the
user-facing interface; the standalone HTML page from the nested app archive is
not served here.

## Run

Install the requirements in a Python environment supported by TensorFlow:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

By default the service listens only on `127.0.0.1:5000`. Configure
`MODEL_HOST` and `MODEL_PORT` when running the Flask app directly. The root
`npm run dev` command starts it automatically; set `MODEL_PYTHON` to the
virtual-environment interpreter if `python3` does not select that environment.

## Endpoints

- `GET /health` reports the model name and class labels.
- `POST /predict` accepts a multipart `image` and returns the highest-ranked
  label, raw score, uncertainty flag, and scores for all four labels.

Images are EXIF-oriented, converted to RGB, resized to 224×224, then passed as
float pixels to the saved model. The saved model contains the training-time
input rescaling layers. The service flags a top score below 0.60 as uncertain.

## Supplied assets

- `app.py`: local inference service adapted from the supplied model app for the
  existing React/Node application. The archive's separate HTML UI and its
  unsupported per-class explanatory text are not used.
- `classes.json`: original output label order.
- `model/apple_disease_mobilenetv2.keras`: supplied trained model.
- `preprocess.py`, `train_model.py`, `evaluate_model.py`: supplied source for
  preprocessing, training, and evaluation.
- `evaluation/metrics_summary.json` and `evaluation/classification_report.txt`:
  supplied evaluation summaries.
- `requirements.txt`: runtime dependencies; `training-requirements.txt` lists
  dependencies for reproducing the training/evaluation workflow.

The same model and scripts appeared in repeated exports. The existing
`preprocess.py`, `classes.json`, training/evaluation scripts, and `.keras`
model match the supplied archive members by SHA-256, so duplicate copies were
not added. The larger `processed_dataset.zip`, sample NumPy arrays, and image
attachments were intentionally not extracted.

## Evaluation and limitations

The supplied evaluation report records 55.6% accuracy on 63 held-out test
images. `SCAB LEAVES` recall was 33%. The score is a raw model output, not a
calibrated probability. This classifier does not verify apple-leaf identity,
image quality, visible symptoms, causes, or treatment. It always ranks one of
its four classes, so results are screening suggestions, not diagnoses.
