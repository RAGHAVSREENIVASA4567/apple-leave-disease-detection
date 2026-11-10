import { CheckCircle2, CircleHelp, Leaf, TriangleAlert } from "lucide-react";

const labels = {
  healthy: "Healthy",
  potential_disease: "Potential disease",
  uncertain: "Needs another look",
  not_apple_leaf: "Not an apple leaf",
};

export default function StatusBadge({ status }) {
  const Icon = status === "healthy" ? CheckCircle2 : status === "potential_disease" ? TriangleAlert : status === "not_apple_leaf" ? Leaf : CircleHelp;
  return <span className={`status-badge status-${status || "uncertain"}`}><Icon size={14} />{labels[status] || "Uncertain"}</span>;
}
