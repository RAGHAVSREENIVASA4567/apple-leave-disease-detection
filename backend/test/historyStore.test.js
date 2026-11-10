import test from "node:test";
import assert from "node:assert/strict";
import sharp from "sharp";
import { deleteScan, findScan, listScans, saveScan, setMongoConnected } from "../services/historyStore.js";

test("stores the original image and serves thumbnails in history lists", async () => {
  setMongoConnected(false);
  const image = await sharp({
    create: {
      width: 16,
      height: 16,
      channels: 3,
      background: { r: 36, g: 112, b: 61 },
    },
  }).png().toBuffer();
  const analysis = {
    health_status: "healthy",
    predicted_class: { label: "HEALTHY LEAVES", title: "Healthy Leaves" },
    class_scores: [
      { label: "HEALTHY LEAVES", title: "Healthy Leaves", score: 0.8 },
      { label: "APPLE ROT LEAVES", title: "Apple Rot Leaves", score: 0.1 },
      { label: "LEAF BLOTCH", title: "Leaf Blotch", score: 0.05 },
      { label: "SCAB LEAVES", title: "Scab Leaves", score: 0.05 },
    ],
    confidence_score: 0.8,
    is_apple_leaf: null,
    image_quality: "not_assessed",
  };

  const saved = await saveScan({
    filename: "leaf.png",
    mimeType: "image/png",
    image,
    analysis,
    provider: "local",
    model: "test-model",
  });

  assert.match(saved.imageData, /^data:image\/png;base64,/);
  assert.equal((await findScan(saved.id)).imageData, saved.imageData);
  const summary = (await listScans()).find((scan) => scan.id === saved.id);
  assert.match(summary.imageData, /^data:image\/jpeg;base64,/);
  assert.notEqual(summary.imageData, saved.imageData);
  assert.equal(summary.analysis.health_status, "healthy");
  assert.equal(await deleteScan(saved.id), true);
  assert.equal(await findScan(saved.id), null);
});
