import mongoose from "mongoose";

const scanSchema = new mongoose.Schema(
  {
    filename: { type: String, required: true },
    mimeType: { type: String, required: true },
    imageData: { type: String, required: true },
    previewData: { type: String, required: true },
    analysis: { type: mongoose.Schema.Types.Mixed, required: true },
    provider: { type: String, required: true },
    model: { type: String, required: true },
  },
  { timestamps: true },
);

export const Scan = mongoose.models.Scan || mongoose.model("Scan", scanSchema);
