import test from "node:test";
import assert from "node:assert/strict";
import sharp from "sharp";
import { analyzeLeaf } from "../controllers/analysisController.js";
import { deleteScan, setMongoConnected } from "../services/historyStore.js";

const modelPrediction = {
  label: "LEAF BLOTCH",
  title: "Leaf Blotch",
  confidence: 0.7,
  uncertain: false,
  probabilities: [
    { label: "LEAF BLOTCH", title: "Leaf Blotch", p: 0.7 },
    { label: "APPLE ROT LEAVES", title: "Apple Rot Leaves", p: 0.1 },
    { label: "HEALTHY LEAVES", title: "Healthy Leaves", p: 0.1 },
    { label: "SCAB LEAVES", title: "Scab Leaves", p: 0.1 },
  ],
};

test("saves the local classifier result through the existing analysis controller", async () => {
  const image = await sharp({
    create: {
      width: 16,
      height: 16,
      channels: 3,
      background: { r: 36, g: 112, b: 61 },
    },
  }).png().toBuffer();
  const previousFetch = globalThis.fetch;
  const previousModelUrl = process.env.MODEL_SERVICE_URL;
  process.env.MODEL_SERVICE_URL = "http://127.0.0.1:5000";
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "http://127.0.0.1:5000/predict");
    const upload = options.body.get("image");
    assert.equal(upload.type, "image/png");
    assert.equal(upload.size, image.length);
    return new Response(JSON.stringify(modelPrediction), {
      headers: { "content-type": "application/json" },
    });
  };
  setMongoConnected(false);

  let responseBody;
  const response = {
    statusCode: 200,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(body) {
      responseBody = body;
      return this;
    },
  };
  let nextError;

  try {
    await analyzeLeaf({
      file: { buffer: image, mimetype: "image/png", originalname: "leaf.png" },
    }, response, (error) => { nextError = error; });

    assert.equal(nextError, undefined);
    assert.equal(response.statusCode, 201);
    assert.equal(responseBody.scan.provider, "local");
    assert.equal(responseBody.scan.model, "apple_disease_mobilenetv2");
    assert.equal(responseBody.scan.analysis.predicted_class.label, "LEAF BLOTCH");
    assert.equal(responseBody.scan.analysis.class_scores[0].score, 0.7);
    assert.equal("possible_causes" in responseBody.scan.analysis, false);
    await deleteScan(responseBody.scan.id);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousModelUrl === undefined) delete process.env.MODEL_SERVICE_URL;
    else process.env.MODEL_SERVICE_URL = previousModelUrl;
  }
});
