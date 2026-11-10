import { validateAnalysis } from "../config/analysisSchema.js";
import { analyzeLeafImage, MODEL_NAME } from "../services/modelService.js";
import { deleteScan, findScan, listScans, saveScan } from "../services/historyStore.js";
import { validateImageBuffer } from "../middleware/upload.js";

export async function analyzeLeaf(request, response, next) {
  try {
    if (!request.file) {
      response.status(400).json({ error: "Select an image before starting the analysis." });
      return;
    }
    const mimeType = await validateImageBuffer(request.file);
    const rawAnalysis = await analyzeLeafImage({
      image: request.file.buffer,
      mimeType,
      filename: request.file.originalname.slice(0, 180) || "apple-leaf-photo",
    });
    const analysis = validateAnalysis(rawAnalysis);
    const scan = await saveScan({
      filename: request.file.originalname.slice(0, 180) || "apple-leaf-photo",
      mimeType,
      image: request.file.buffer,
      analysis,
      provider: "local",
      model: MODEL_NAME,
    });

    response.status(201).json({
      scan,
      historyPersistence: process.env.MONGODB_URI ? "mongodb" : "memory",
    });
  } catch (error) {
    if (error.code === "INVALID_MODEL_PREDICTION") {
      error.status = 502;
      error.message = "The local model returned an invalid prediction. Please retry.";
    }
    next(error);
  }
}

export async function getHistory(_request, response, next) {
  try {
    response.json({ scans: await listScans() });
  } catch (error) {
    next(error);
  }
}

export async function getHistoryItem(request, response, next) {
  try {
    const scan = await findScan(request.params.id);
    if (!scan) {
      response.status(404).json({ error: "Analysis not found." });
      return;
    }
    response.json({ scan });
  } catch (error) {
    next(error);
  }
}

export async function deleteHistoryItem(request, response, next) {
  try {
    const deleted = await deleteScan(request.params.id);
    if (!deleted) {
      response.status(404).json({ error: "Analysis not found." });
      return;
    }
    response.status(204).end();
  } catch (error) {
    next(error);
  }
}
