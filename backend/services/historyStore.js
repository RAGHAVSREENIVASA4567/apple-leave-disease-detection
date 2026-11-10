import { randomUUID } from "node:crypto";
import sharp from "sharp";
import { Scan } from "../models/Scan.js";

const memoryScans = new Map();
const MEMORY_HISTORY_LIMIT = 20;
let mongoConnected = false;

export function setMongoConnected(value) {
  mongoConnected = value;
}

function toPublicScan(scan) {
  return {
    id: String(scan._id ?? scan.id),
    filename: scan.filename,
    mimeType: scan.mimeType,
    imageData: scan.imageData,
    analysis: scan.analysis,
    provider: scan.provider,
    model: scan.model,
    createdAt: scan.createdAt,
  };
}

function toHistorySummary(scan) {
  const { imageData, ...metadata } = toPublicScan(scan);
  return { ...metadata, imageData: scan.previewData };
}

export async function saveScan({ filename, mimeType, image, analysis, provider, model }) {
  const preview = await sharp(image)
    .rotate()
    .resize(480, 480, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 72 })
    .toBuffer();
  const record = {
    filename,
    mimeType,
    imageData: `data:${mimeType};base64,${image.toString("base64")}`,
    previewData: `data:image/jpeg;base64,${preview.toString("base64")}`,
    analysis,
    provider,
    model,
    createdAt: new Date(),
  };

  if (mongoConnected) {
    const saved = await Scan.create(record);
    return toPublicScan(saved);
  }

  const id = randomUUID();
  memoryScans.set(id, { ...record, id });
  if (memoryScans.size > MEMORY_HISTORY_LIMIT) {
    memoryScans.delete(memoryScans.keys().next().value);
  }
  return toPublicScan(memoryScans.get(id));
}

export async function listScans() {
  if (mongoConnected) {
    const scans = await Scan.find()
      .select("-imageData")
      .sort({ createdAt: -1 })
      .limit(200)
      .lean();
    return scans.map(toHistorySummary);
  }
  return [...memoryScans.values()]
    .sort((a, b) => b.createdAt - a.createdAt)
    .map(toHistorySummary);
}

export async function findScan(id) {
  if (mongoConnected) {
    if (!/^[a-f\d]{24}$/i.test(id)) return null;
    const scan = await Scan.findById(id).lean();
    return scan ? toPublicScan(scan) : null;
  }
  const scan = memoryScans.get(id);
  return scan ? toPublicScan(scan) : null;
}

export async function deleteScan(id) {
  if (mongoConnected) {
    if (!/^[a-f\d]{24}$/i.test(id)) return false;
    const result = await Scan.findByIdAndDelete(id);
    return Boolean(result);
  }
  return memoryScans.delete(id);
}
