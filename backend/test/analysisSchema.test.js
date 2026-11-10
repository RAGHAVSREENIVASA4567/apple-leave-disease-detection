import test from "node:test";
import assert from "node:assert/strict";
import { validateAnalysis } from "../config/analysisSchema.js";

function prediction({
  label = "APPLE ROT LEAVES",
  confidence = 0.7,
  uncertain = false,
  probabilities = [
    { label: "APPLE ROT LEAVES", p: 0.7 },
    { label: "HEALTHY LEAVES", p: 0.1 },
    { label: "LEAF BLOTCH", p: 0.1 },
    { label: "SCAB LEAVES", p: 0.1 },
  ],
} = {}) {
  return { label, title: label, confidence, uncertain, probabilities };
}

test("keeps all model scores while limiting the analysis to actual classifier output", () => {
  const analysis = validateAnalysis(prediction());
  assert.deepEqual(analysis.predicted_class, {
    label: "APPLE ROT LEAVES",
    title: "Apple Rot Leaves",
  });
  assert.deepEqual(analysis.class_scores.map(({ label }) => label), [
    "APPLE ROT LEAVES",
    "HEALTHY LEAVES",
    "LEAF BLOTCH",
    "SCAB LEAVES",
  ]);
  assert.equal(analysis.confidence_score, 0.7);
  assert.equal(analysis.health_status, "potential_disease");
  assert.equal(analysis.is_apple_leaf, null);
  assert.equal(analysis.image_quality, "not_assessed");
  assert.equal("visual_symptoms" in analysis, false);
  assert.equal("possible_causes" in analysis, false);
  assert.equal("recommendations" in analysis, false);
});

test("marks low top-class scores uncertain without discarding the classifier output", () => {
  const analysis = validateAnalysis(prediction({
    confidence: 0.45,
    uncertain: true,
    probabilities: [
      { label: "SCAB LEAVES", p: 0.45 },
      { label: "HEALTHY LEAVES", p: 0.25 },
      { label: "LEAF BLOTCH", p: 0.2 },
      { label: "APPLE ROT LEAVES", p: 0.1 },
    ],
    label: "SCAB LEAVES",
  }));
  assert.equal(analysis.health_status, "uncertain");
  assert.equal(analysis.predicted_class.label, "SCAB LEAVES");
});

test("marks a high-scoring healthy class without inventing image verification", () => {
  const analysis = validateAnalysis(prediction({
    label: "HEALTHY LEAVES",
    confidence: 0.8,
    probabilities: [
      { label: "HEALTHY LEAVES", p: 0.8 },
      { label: "APPLE ROT LEAVES", p: 0.1 },
      { label: "LEAF BLOTCH", p: 0.05 },
      { label: "SCAB LEAVES", p: 0.05 },
    ],
  }));
  assert.equal(analysis.health_status, "healthy");
  assert.equal(analysis.is_apple_leaf, null);
});

test("rejects unknown classes, missing scores, and inconsistent top-class metadata", () => {
  assert.throws(() => validateAnalysis(prediction({ label: "APPLE SCAB" })), /expected class scores/);
  assert.throws(() => validateAnalysis(prediction({
    probabilities: [{ label: "APPLE ROT LEAVES", p: 1 }],
  })), /expected class scores/);
  assert.throws(() => validateAnalysis(prediction({ uncertain: true })), /inconsistent top-class/);
  assert.throws(() => validateAnalysis(prediction({
    probabilities: [
      { label: "APPLE ROT LEAVES", p: 0.8 },
      { label: "HEALTHY LEAVES", p: 0.1 },
      { label: "LEAF BLOTCH", p: 0.1 },
      { label: "SCAB LEAVES", p: 0.1 },
    ],
  })), /probability distribution/);
});
