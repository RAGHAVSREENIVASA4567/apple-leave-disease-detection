export const MODEL_CLASS_NAMES = [
  "APPLE ROT LEAVES",
  "HEALTHY LEAVES",
  "LEAF BLOTCH",
  "SCAB LEAVES",
];
const CLASS_TITLES = {
  "APPLE ROT LEAVES": "Apple Rot Leaves",
  "HEALTHY LEAVES": "Healthy Leaves",
  "LEAF BLOTCH": "Leaf Blotch",
  "SCAB LEAVES": "Scab Leaves",
};
const CLASS_NAMES = MODEL_CLASS_NAMES;
const UNCERTAIN_BELOW = 0.60;
const MAX_SCORE_SUM_ERROR = 0.01;

function invalidAnalysis(message) {
  const error = new Error(message);
  error.code = "INVALID_MODEL_PREDICTION";
  return error;
}

export function validateAnalysis(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw invalidAnalysis("The model response must be a JSON object.");
  }

  const { label, confidence, uncertain, probabilities } = value;
  if (!CLASS_NAMES.includes(label) || !Array.isArray(probabilities) ||
      probabilities.length !== CLASS_NAMES.length) {
    throw invalidAnalysis("The model response did not contain the expected class scores.");
  }

  const seen = new Set();
  const classScores = probabilities.map((item) => {
    if (!item || !CLASS_NAMES.includes(item.label) || seen.has(item.label) ||
        typeof item.p !== "number" || !Number.isFinite(item.p) || item.p < 0 || item.p > 1) {
      throw invalidAnalysis("The model response contains an invalid class score.");
    }
    seen.add(item.label);
    return { label: item.label, title: CLASS_TITLES[item.label], score: item.p };
  });

  if (CLASS_NAMES.some((className) => !seen.has(className))) {
    throw invalidAnalysis("The model response is missing a class score.");
  }

  const scoreSum = classScores.reduce((sum, item) => sum + item.score, 0);
  if (Math.abs(scoreSum - 1) > MAX_SCORE_SUM_ERROR) {
    throw invalidAnalysis("The model class scores do not form a probability distribution.");
  }

  classScores.sort((left, right) => right.score - left.score);
  const highest = classScores[0];
  if (label !== highest.label || typeof confidence !== "number" ||
      !Number.isFinite(confidence) || Math.abs(confidence - highest.score) > 1e-6 ||
      typeof uncertain !== "boolean" || uncertain !== (highest.score < UNCERTAIN_BELOW)) {
    throw invalidAnalysis("The model response has inconsistent top-class information.");
  }

  const isUncertain = highest.score < UNCERTAIN_BELOW;
  const healthStatus = isUncertain
    ? "uncertain"
    : highest.label === "HEALTHY LEAVES"
      ? "healthy"
      : "potential_disease";

  return {
    predicted_class: { label: highest.label, title: highest.title },
    class_scores: classScores,
    confidence_score: highest.score,
    health_status: healthStatus,
    uncertain: isUncertain,
    is_apple_leaf: null,
    image_quality: "not_assessed",
    uncertainty_notes: "This classifier always ranks one of its four training labels. It cannot confirm that a photo shows an apple leaf or assess image quality, symptoms, causes, or treatment. The score is not calibrated; the supplied held-out evaluation was correct on 55.6% of 63 images, with 33% recall for SCAB LEAVES. Treat this as a screening aid, not a diagnosis.",
  };
}
