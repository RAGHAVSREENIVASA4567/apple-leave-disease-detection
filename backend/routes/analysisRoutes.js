import { Router } from "express";
import { analyzeLeaf, deleteHistoryItem, getHistory, getHistoryItem } from "../controllers/analysisController.js";

const router = Router();

router.post("/analyze-leaf", analyzeLeaf);
router.get("/history", getHistory);
router.get("/history/:id", getHistoryItem);
router.delete("/history/:id", deleteHistoryItem);

export default router;
