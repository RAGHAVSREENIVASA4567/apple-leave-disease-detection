# Apple Leaf Disease Check: deployment (step 5)

A small web app: a farmer or field officer takes or uploads a photo of an apple leaf, and the
trained MobileNetV2 model returns (steps 5 and 6 of the project flow):

- the **disease name** (Apple rot, Leaf blotch, Apple scab, or "No disease found")
- a **confidence score** and the probability of every class
- the **likely cause**
- **suggested remedy** steps

When confidence is under 60%, the app labels the result "Possibly: ...", shows a "not sure" warning,
and words the remedy as "if this result is correct".

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

**On a phone:** connect the phone to the same Wi-Fi as the computer, then open
`http://<computer's IP address>:5000`. "Take photo" opens the phone camera directly.
(Windows: `ipconfig`; macOS/Linux: `ifconfig` or `ip a`. Allow Python through the firewall if asked.)

If the model fails to load, install the versions it was trained with: `pip install tensorflow==2.21.0`.

## API

```bash
curl -F "image=@leaf.jpg" http://localhost:5000/predict
```

Returns JSON: `label`, `title`, `about`, `confidence`, `uncertain`, and `probabilities` (sorted).
`GET /health` returns `{"status": "ok"}`.

## Configuration (top of app.py)

- `UNCERTAIN_BELOW`: confidence under which the app shows a "not sure" warning (default 0.60)
- `CLASS_INFO`: the disease name, cause and remedy text shown per class. Have your faculty guide or a
  horticulture officer review this before any real field use.
- `MAX_UPLOAD_MB`: upload size limit (default 10)

## Limitations (please read before showing this to real users)

- **Leaves only.** The model was trained on leaf photos. It has never seen fruit, and it will still
  return an answer for any image, including non-apple or non-leaf photos.
- **Accuracy is modest:** 55.6% on the held-out test set (95% CI about 43-67%, only 63 images). Scab is
  the weakest class (33% recall).
- **Confidence is not reliable.** In testing, some scab leaves were called Healthy at 96-97% confidence,
  so a high percentage does not mean the answer is right. This is why the app always shows a
  "prototype, not a diagnosis" notice.
- The built-in Flask server is for demos. For wider use, run behind a production server
  (`waitress-serve --port=5000 app:app` on Windows, `gunicorn app:app` on Linux) with HTTPS.

## About the cause and remedy text

- It is **general guidance** compiled from university extension and SKUAST-Kashmir material (Sources below),
  not a prescription. It deliberately gives **no fungicide product names or doses**: registrations and the
  official spray schedule differ by region, so the app points users to the Horticulture Department / SKUAST-K
  schedule instead.
- **"Apple rot" is ambiguous.** The dataset labels this class only as "Apple Rot" and does not name the
  pathogen. The app's wording (black rot / frogeye leaf spot, *Botryosphaeria obtusa*) is the most common
  rot-type disease on apple *leaves*, but it is an assumption. Confirm what the images actually show with
  your faculty guide or the dataset's source before presenting it as fact.
- The urea-spray advice for scab (3-4% at 15-20% leaf yellowing/fall) comes from a SKUAST-Kashmir advisory as
  reported in the press; confirm it is still current. Other sources quote different rates.
- Black rot and leaf blotch guidance comes mostly from non-Kashmir sources (US/European extension and ICAR
  trials). Scab guidance has Kashmir-specific sources.

## Sources

- SKUAST-Kashmir urea advisory (Greater Kashmir): https://www.greaterkashmir.com/?p=168660
- SKUAST experts on scab sanitation and spray timing (Down To Earth): https://www.downtoearth.org.in/agriculture/fungal-scab-attack-can-spell-doom-for-kashmir-s-apple-growers-this-year-74631
- Kashmir spray schedule and scab management (Kashmir Reader): https://kashmirreader.com/?p=194402
- Apple scab (Cornell): https://bpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/d/11253/files/2024/12/AppleScab.pdf
- Marssonina blotch (Connecticut Agricultural Experiment Station): https://portal.ct.gov/-/media/caes/documents/publications/fact_sheets/plant_pathology_and_ecology/2019/marssonina-blotch-of-apple.pdf
- Apple blotch disease (Virginia Tech, SPES-509): https://treefruitpathology.spes.vt.edu/wp-content/uploads/2023/06/SPES-509.pdf
- Marssonina fungicide trials in India (ICAR, Sharma & Kumar): https://agris.fao.org/search/en/records/64747342bf943c8c7982ff24
- Black rot / frogeye leaf spot (Michigan State): https://www.canr.msu.edu/ipm/diseases/black_rot_blossom_end_rot_frogeye_leaf_spot
- Black rot / frogeye leaf spot (NC State): https://apples.ces.ncsu.edu/2018/03/leaf-spots-of-bother-the-troublesome-3
- Black rot in apple (University of Minnesota): https://extension.umn.edu/plant-diseases/black-rot-apple

## Files

- `app.py`: server, preprocessing, prediction, and the web page
- `model/apple_disease_mobilenetv2.keras`: trained model from step 3
- `requirements.txt`
