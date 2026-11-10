import test from "node:test";
import assert from "node:assert/strict";
import { MODEL_CLASS_NAMES } from "../config/analysisSchema.js";
import { analyzeLeafImage, isModelServiceReady, MODEL_NAME } from "../services/modelService.js";

function response(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("sends the selected image to the local model service", async () => {
  const previousFetch = globalThis.fetch;
  let requestUrl;
  let requestOptions;
  globalThis.fetch = async (url, options) => {
    requestUrl = url;
    requestOptions = options;
    return response({ label: "HEALTHY LEAVES" });
  };

  try {
    const result = await analyzeLeafImage({
      image: Buffer.from("image bytes"),
      mimeType: "image/png",
      filename: "leaf.png",
    });
    assert.equal(requestUrl, "http://127.0.0.1:5000/predict");
    assert.equal(requestOptions.method, "POST");
    const upload = requestOptions.body.get("image");
    assert.equal(upload.name, "leaf.png");
    assert.equal(upload.type, "image/png");
    assert.equal(await upload.text(), "image bytes");
    assert.deepEqual(result, { label: "HEALTHY LEAVES" });
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("reports the exact local model readiness and never falls back to a hosted provider", async () => {
  const previousFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (url) => {
    calls += 1;
    assert.equal(url, "http://127.0.0.1:5000/health");
    return response({ status: "ok", model: MODEL_NAME, classes: MODEL_CLASS_NAMES });
  };

  try {
    assert.equal(await isModelServiceReady(), true);
    globalThis.fetch = async () => {
      calls += 1;
      throw new Error("offline");
    };
    await assert.rejects(
      analyzeLeafImage({ image: Buffer.from("image"), mimeType: "image/jpeg" }),
      /local MobileNetV2 model service is unavailable/,
    );
    assert.equal(calls, 2);
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("returns not-ready when the model name or class order does not match", async () => {
  const previousFetch = globalThis.fetch;
  let health = {
    status: "ok",
    model: "another-model",
    classes: MODEL_CLASS_NAMES,
  };
  globalThis.fetch = async () => response(health);
  try {
    assert.equal(await isModelServiceReady(), false);
    health = {
      status: "ok",
      model: MODEL_NAME,
      classes: [...MODEL_CLASS_NAMES].reverse(),
    };
    assert.equal(await isModelServiceReady(), false);
  } finally {
    globalThis.fetch = previousFetch;
  }
});
