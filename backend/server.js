import "dotenv/config";
import cors from "cors";
import express from "express";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import multer from "multer";
import mongoose from "mongoose";
import analysisRoutes from "./routes/analysisRoutes.js";
import { imageUpload } from "./middleware/upload.js";
import { isModelServiceReady, MODEL_NAME } from "./services/modelService.js";
import { setMongoConnected } from "./services/historyStore.js";

const app = express();
const port = Number(process.env.PORT || 4000);
const allowedOrigins = (process.env.CLIENT_ORIGIN || "http://localhost:5173,http://127.0.0.1:5173")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

app.disable("x-powered-by");
app.use(helmet());
app.use(cors({ origin: allowedOrigins }));
app.use(express.json({ limit: "32kb" }));
app.use("/api", rateLimit({ windowMs: 60_000, limit: 30, standardHeaders: "draft-8", legacyHeaders: false }));
app.use("/api", (request, response, next) => {
  if (request.path === "/analyze-leaf" && request.method === "POST") {
    imageUpload(request, response, (error) => {
      if (error) {
        if (error instanceof multerError) {
          response.status(400).json({
            error: error.code === "LIMIT_FILE_SIZE"
              ? `Image exceeds the ${process.env.MAX_UPLOAD_MB || 8} MB upload limit.`
              : "Upload one image at a time.",
          });
          return;
        }
        response.status(400).json({ error: error.message });
        return;
      }
      next();
    });
    return;
  }
  next();
});
app.use("/api", analysisRoutes);

app.get("/api/health", async (_request, response) => {
  const modelReady = await isModelServiceReady();
  response.json({
    status: "ok",
    provider: "local",
    model: MODEL_NAME,
    modelReady,
    historyPersistence: process.env.MONGODB_URI ? "mongodb" : "memory",
  });
});

app.use((error, _request, response, _next) => {
  const status = Number.isInteger(error.status) ? error.status : 500;
  if (status >= 500) console.error("API request failed:", error.message);
  response.status(status).json({
    error: status >= 500 && status === 500
      ? "The analysis service encountered an error. Check the backend configuration and try again."
      : error.message,
  });
});

async function start() {
  if (process.env.MONGODB_URI) {
    try {
      await mongoose.connect(process.env.MONGODB_URI);
      setMongoConnected(true);
      console.log("Connected to MongoDB; scan history is persistent.");
    } catch (error) {
      console.error("MongoDB connection failed:", error.message);
      process.exitCode = 1;
      return;
    }
  } else {
    console.warn("MONGODB_URI is not set; scan history will be held in memory and cleared on restart.");
  }

  app.listen(port, () => {
    console.log(`AppleCare AI API listening on http://localhost:${port}`);
  });
}

const multerError = multer.MulterError;
start();
