"""
Step "5. Deployment": a farmer / field officer uploads a new leaf photo via the app
and gets the predicted apple leaf condition from the trained MobileNetV2 model.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 (or http://<this-computer's-IP>:5000 from a phone
on the same Wi-Fi -- the phone's camera can be used directly from the page).

API (for other apps / testing):
    POST /predict   multipart form, field "image"  ->  JSON prediction
    GET  /health                                   ->  {"status": "ok"}
"""

import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageOps, UnidentifiedImageError

# ----------------------------- Config -----------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "apple_disease_mobilenetv2.keras")
IMG_SIZE = (224, 224)                # same size used in preprocessing/training
MAX_UPLOAD_MB = 10
UNCERTAIN_BELOW = 0.60               # below this top-class probability, flag as uncertain

# Order MUST match training (alphabetical folder names from image_dataset_from_directory).
CLASS_NAMES = ["APPLE ROT LEAVES", "HEALTHY LEAVES", "LEAF BLOTCH", "SCAB LEAVES"]

# Plain-language output shown with each result: what it is, the likely cause, and
# suggested remedy steps. Content is drawn from university extension / SKUAST-Kashmir
# material (see README "Sources"). It is general guidance only: fungicide products, doses
# and timing are deliberately NOT given here because registrations and the official spray
# schedule differ by region -- farmers are pointed to the local Horticulture Department /
# SKUAST-K schedule instead. Have your faculty guide / a horticulture officer review it.
CLASS_INFO = {
    "APPLE ROT LEAVES": {
        "title": "Apple rot",
        "about": "Rot-type lesions on the leaf, caused by a fungal infection.",
        "cause": (
            "A fungal rot disease. On leaves this is most often black rot / frogeye leaf spot "
            "(fungus Botryosphaeria obtusa), seen as round tan spots with purple edges. The fungus "
            "survives the winter in dead branches, cankers and shriveled fruit left on or under the tree, "
            "and spreads in warm, wet weather."
        ),
        "remedy": [
            "Collect and destroy fallen infected leaves and any shriveled (mummified) fruit; burn or bury them.",
            "In the dormant season, prune out dead branches and sunken cankers, cutting well below the damaged wood, and burn or bury the prunings.",
            "Avoid wounding the tree and keep it healthy and well watered; open wounds let the fungus in.",
            "Ask your horticulture officer for the recommended fungicide and timing in your area, and follow the product label.",
        ],
    },
    "HEALTHY LEAVES": {
        "title": "No disease found",
        "about": "The model did not find a clear disease pattern on this leaf. This is not a guarantee that the leaf is healthy.",
        "cause": "No disease cause identified from this photo.",
        "remedy": [
            "Check the leaf yourself. If you can see spots, curling, yellowing or unusual patches, treat this result as wrong and ask a field officer to look at it.",
            "This tool is not reliable at ruling out disease: in testing, nearly half of the leaves it called healthy actually had a disease.",
            "Keep up routine orchard care: clear fallen leaves in autumn and check leaves regularly during wet weather.",
        ],
    },
    "LEAF BLOTCH": {
        "title": "Leaf blotch",
        "about": "Brown or purplish blotches on the leaf; can cause early leaf fall.",
        "cause": (
            "Marssonina leaf blotch, a fungal disease (Diplocarpon coronariae, also called Marssonina coronaria). "
            "The fungus survives the winter in fallen leaves, and spores spread in warm, rainy weather."
        ),
        "remedy": [
            "Collect and destroy fallen leaves in autumn (burn or bury); this removes the source of next season's infection.",
            "Prune so air moves through the canopy and leaves dry faster after rain.",
            "A urea spray on fallen leaves can speed up their decay; ask your horticulture officer about the right rate and timing.",
            "Follow the Horticulture Department / SKUAST-K spray schedule for fungicide choice and timing. Not every fungicide is approved for this disease everywhere.",
        ],
    },
    "SCAB LEAVES": {
        "title": "Apple scab",
        "about": "Olive-brown scabby spots on the leaf; heavily infected leaves curl and drop early.",
        "cause": (
            "A fungal disease caused by Venturia inaequalis. The fungus survives the winter in fallen infected leaves; "
            "in spring, rain releases spores that infect young leaves and fruit. Cool, wet weather and long periods of leaf wetness make it worse."
        ),
        "remedy": [
            "Collect and destroy fallen leaves in autumn (burn or bury). This is one of the most effective steps.",
            "SKUAST-Kashmir has advised a 3-4% urea spray once about 15-20% of leaves have yellowed or fallen, to speed up leaf decay. Confirm the current advice with your KVK or horticulture officer.",
            "Prune to open the canopy so leaves dry faster.",
            "Protect new leaves and fruit with well-timed fungicide sprays from early leaf stage through blossom and fruit development, following the Horticulture Department's Kashmir spray schedule. Do not mix products other than those recommended.",
        ],
    },
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
# Warm-up so the first real request isn't slow
model(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), training=False)
print("Model ready.")


def preprocess(file_bytes: bytes) -> np.ndarray:
    """Same steps as preprocess.py: respect EXIF rotation, RGB, bilinear resize to 224x224.
    The model itself contains the 1/255 and [-1,1] rescaling layers, so pixels stay 0-255."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Photo is too large (max {MAX_UPLOAD_MB} MB)."), 413


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No photo received. Please choose or take a photo."), 400
    try:
        x = preprocess(file.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file isn't a readable image. Please upload a JPG or PNG photo."), 400

    probs = model(x, training=False).numpy()[0]
    order = np.argsort(-probs)
    top = CLASS_NAMES[order[0]]
    confidence = float(probs[order[0]])

    return jsonify(
        label=top,
        title=CLASS_INFO[top]["title"],
        about=CLASS_INFO[top]["about"],
        cause=CLASS_INFO[top]["cause"],
        remedy=CLASS_INFO[top]["remedy"],
        confidence=confidence,
        uncertain=confidence < UNCERTAIN_BELOW,
        probabilities=[
            {"label": CLASS_NAMES[i], "title": CLASS_INFO[CLASS_NAMES[i]]["title"], "p": float(probs[i])}
            for i in order
        ],
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple Leaf Disease Check</title>
<style>
  :root { --green:#1b5e20; --green-soft:#e8f5e9; --ink:#1c1c1c; --muted:#5f6b60;
          --line:#d5ddd6; --warn-bg:#fff4e0; --warn-ink:#7a4a00; --bad:#b3261e; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         color:var(--ink); background:#f6f8f6; }
  header { background:var(--green); color:#fff; padding:18px 16px; }
  header h1 { margin:0; font-size:1.25rem; }
  header p { margin:4px 0 0; font-size:.9rem; opacity:.9; }
  main { max-width:560px; margin:0 auto; padding:16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; margin-bottom:14px; }
  .tips { font-size:.9rem; color:var(--muted); margin:0 0 12px; }
  .buttons { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  button.pick { font-size:1rem; font-weight:600; padding:16px 8px; border-radius:12px; cursor:pointer;
                border:2px solid var(--green); background:var(--green); color:#fff; }
  button.pick.alt { background:#fff; color:var(--green); }
  button.pick:focus-visible { outline:3px solid #66bb6a; outline-offset:2px; }
  #preview { display:none; width:100%; max-height:340px; object-fit:contain; border-radius:10px;
             background:#eee; margin-top:14px; }
  #status { display:none; margin-top:14px; font-weight:600; color:var(--green); }
  #error { display:none; margin-top:14px; padding:12px; border-radius:10px; background:#fdecea; color:var(--bad); }
  #result { display:none; }
  .verdict { font-size:1.5rem; font-weight:700; margin:0; }
  .conf { color:var(--muted); margin:2px 0 10px; }
  .about { margin:0 0 12px; }
  .uncertain { display:none; background:var(--warn-bg); color:var(--warn-ink); padding:10px 12px;
               border-radius:10px; margin-bottom:12px; font-weight:600; }
  h2.sec { font-size:1rem; margin:16px 0 6px; color:var(--green); }
  .cause { margin:0 0 4px; }
  ol.remedy { margin:0; padding-left:20px; }
  ol.remedy li { margin:0 0 8px; }
  .row { display:grid; grid-template-columns:110px 1fr 46px; gap:8px; align-items:center; margin:6px 0; font-size:.9rem; }
  .bar { height:10px; background:var(--green-soft); border-radius:6px; overflow:hidden; }
  .bar > span { display:block; height:100%; background:var(--green); }
  .row.top .bar > span { background:#2e7d32; }
  .disclaimer { font-size:.82rem; color:var(--muted); border-top:1px solid var(--line); padding-top:10px; margin-top:12px; }
  footer { text-align:center; color:var(--muted); font-size:.8rem; padding:8px 16px 24px; }
</style>
</head>
<body>
<header>
  <h1>Apple Leaf Disease Check</h1>
  <p>Take or upload a photo of an apple leaf to check for common diseases.</p>
</header>
<main>
  <section class="card">
    <p class="tips">For best results: one leaf, in focus, in daylight, filling most of the photo.
       This version checks <strong>leaves only</strong> &mdash; fruit photos are not supported yet.</p>
    <div class="buttons">
      <button class="pick" id="btnCamera" type="button">Take photo</button>
      <button class="pick alt" id="btnGallery" type="button">Choose photo</button>
    </div>
    <input id="inCamera" type="file" accept="image/*" capture="environment" hidden>
    <input id="inGallery" type="file" accept="image/*" hidden>
    <img id="preview" alt="Selected leaf photo">
    <div id="status" role="status">Analyzing photo&hellip;</div>
    <div id="error" role="alert"></div>
  </section>

  <section class="card" id="result" aria-live="polite">
    <p class="verdict" id="verdict"></p>
    <p class="conf" id="conf"></p>
    <div class="uncertain" id="uncertain">The model is not sure about this photo. Retake it in better light, or ask a field officer to check the leaf.</div>
    <p class="about" id="about"></p>
    <h2 class="sec">Likely cause</h2>
    <p class="cause" id="cause"></p>
    <h2 class="sec" id="remedyHead">Suggested remedy</h2>
    <ol class="remedy" id="remedy"></ol>
    <h2 class="sec">All possibilities</h2>
    <div id="bars"></div>
    <p class="disclaimer">This is a prototype screening aid, not a diagnosis. Its predictions can be wrong,
       including when it appears confident. Please confirm with a horticulture / agriculture officer before treating your trees.</p>
  </section>
</main>
<footer>Apple disease detection &middot; MobileNetV2 transfer-learning model</footer>

<script>
const $ = id => document.getElementById(id);
const preview = $("preview"), statusEl = $("status"), errorEl = $("error"), resultEl = $("result");

$("btnCamera").onclick = () => $("inCamera").click();
$("btnGallery").onclick = () => $("inGallery").click();
$("inCamera").onchange = e => handle(e.target.files[0]);
$("inGallery").onchange = e => handle(e.target.files[0]);

async function handle(file) {
  if (!file) return;
  errorEl.style.display = "none";
  resultEl.style.display = "none";
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  statusEl.style.display = "block";
  const form = new FormData();
  form.append("image", file);
  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
    show(data);
  } catch (err) {
    errorEl.textContent = err.message || "Could not reach the app. Check your connection and try again.";
    errorEl.style.display = "block";
  } finally {
    statusEl.style.display = "none";
    // allow re-selecting the same file
    $("inCamera").value = ""; $("inGallery").value = "";
  }
}

function show(d) {
  $("verdict").textContent = (d.uncertain ? "Possibly: " : "") + d.title;
  $("conf").textContent = "Confidence: " + Math.round(d.confidence * 100) + "%";
  $("about").textContent = d.about;
  $("uncertain").style.display = d.uncertain ? "block" : "none";
  $("cause").textContent = d.cause;
  $("remedyHead").textContent = d.uncertain ? "Suggested steps, if this result is correct" : "Suggested remedy";
  const list = $("remedy");
  list.textContent = "";
  d.remedy.forEach(step => { const li = document.createElement("li"); li.textContent = step; list.appendChild(li); });
  const bars = $("bars");
  bars.textContent = "";
  d.probabilities.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === 0 ? " top" : "");
    const name = document.createElement("span"); name.textContent = p.title;
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("span"); fill.style.width = (p.p * 100).toFixed(1) + "%";
    bar.appendChild(fill);
    const pct = document.createElement("span"); pct.textContent = Math.round(p.p * 100) + "%";
    row.append(name, bar, pct);
    bars.appendChild(row);
  });
  resultEl.style.display = "block";
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # host 0.0.0.0 so a phone on the same Wi-Fi can open the app
    app.run(host="0.0.0.0", port=port, debug=False)
