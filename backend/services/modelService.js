import { MODEL_CLASS_NAMES } from "../config/analysisSchema.js";

function modelServiceUrl() {
  return process.env.MODEL_SERVICE_URL || "http://127.0.0.1:5000";
}

function requestTimeoutMs() {
  const timeout = Number(process.env.MODEL_SERVICE_TIMEOUT_MS || 30_000);
  return Number.isFinite(timeout) && timeout > 0 ? timeout : 30_000;
}

export const MODEL_NAME = "apple_disease_mobilenetv2";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${modelServiceUrl().replace(/\/+$/, "")}${path}`, {
      ...options,
      signal: AbortSignal.timeout(requestTimeoutMs()),
    });
  } catch (error) {
    const message = error.name === "TimeoutError"
      ? "The local model took too long to respond. Please retry."
      : "The local MobileNetV2 model service is unavailable. Start the model service and try again.";
    const serviceError = new Error(message);
    serviceError.status = 503;
    throw serviceError;
  }

  let body;
  try {
    body = await response.json();
  } catch {
    const error = new Error("The local model service returned an unreadable response.");
    error.status = 502;
    throw error;
  }

  if (!response.ok) {
    const error = new Error(typeof body.error === "string" ? body.error : "The local model could not analyze this image.");
    error.status = response.status === 413 ? 413 : response.status === 400 ? 400 : 502;
    throw error;
  }
  return body;
}

export async function analyzeLeafImage({ image, mimeType, filename }) {
  const form = new FormData();
  form.append("image", new Blob([image], { type: mimeType }), filename || "leaf-image");
  return request("/predict", { method: "POST", body: form });
}

export async function isModelServiceReady() {
  try {
    const health = await request("/health");
    return health.status === "ok" && health.model === MODEL_NAME &&
      Array.isArray(health.classes) &&
      health.classes.length === MODEL_CLASS_NAMES.length &&
      health.classes.every((className, index) => className === MODEL_CLASS_NAMES[index]);
  } catch {
    return false;
  }
}
