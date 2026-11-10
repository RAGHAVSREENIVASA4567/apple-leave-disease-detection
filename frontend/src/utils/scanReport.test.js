import test from "node:test";
import assert from "node:assert/strict";
import { createScanReportCsv } from "./scanReport.js";

test("creates a quoted CSV report and escapes embedded quotes", () => {
  const csv = createScanReportCsv([{
    createdAt: "2026-11-04T10:00:00.000Z",
    filename: 'leaf, "north".jpg',
    analysis: {
      predicted_class: { label: "SCAB LEAVES", title: "Scab Leaves" },
      health_status: "potential_disease",
      confidence_score: 0.7,
      image_quality: "not_assessed",
    },
    model: "apple_disease_mobilenetv2",
  }]);

  assert.match(csv, /^"Date","Image","Top model class"/);
  assert.match(csv, /"leaf, ""north"".jpg","Scab Leaves"/);
  assert.match(csv, /"70\.0%","not_assessed","apple_disease_mobilenetv2"$/);
});
