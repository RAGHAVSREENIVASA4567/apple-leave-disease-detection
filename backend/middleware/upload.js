import multer from "multer";
import { fileTypeFromBuffer } from "file-type";

const maxUploadMb = Number.parseInt(process.env.MAX_UPLOAD_MB || "8", 10);
const maxUploadBytes = (Number.isFinite(maxUploadMb) && maxUploadMb > 0 ? maxUploadMb : 8) * 1024 * 1024;
const supportedMimeTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

export const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: maxUploadBytes, files: 1 },
  fileFilter(_request, file, callback) {
    if (!supportedMimeTypes.has(file.mimetype)) {
      callback(new Error("Unsupported image format. Upload a JPEG, PNG, or WebP image."));
      return;
    }
    callback(null, true);
  },
}).single("image");

export async function validateImageBuffer(file) {
  const detectedType = await fileTypeFromBuffer(file.buffer);
  if (!detectedType || !supportedMimeTypes.has(detectedType.mime)) {
    const error = new Error("The uploaded file is not a valid JPEG, PNG, or WebP image.");
    error.status = 400;
    throw error;
  }
  if (detectedType.mime !== file.mimetype) {
    const error = new Error("The uploaded image content does not match its declared format.");
    error.status = 400;
    throw error;
  }
  return detectedType.mime;
}
