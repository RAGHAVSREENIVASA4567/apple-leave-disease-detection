const reportColumns = [
  ["Date", (scan) => new Date(scan.createdAt).toISOString()],
  ["Image", (scan) => scan.filename],
  ["Top model class", (scan) => scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || "Uncertain"],
  ["Health status", (scan) => scan.analysis.health_status],
  ["Top-class score (uncalibrated)", (scan) => typeof scan.analysis.confidence_score === "number" ? `${(scan.analysis.confidence_score * 100).toFixed(1)}%` : scan.analysis.confidence || ""],
  ["Image quality", (scan) => scan.analysis.image_quality],
  ["Model", (scan) => scan.model],
];

function escapeCsv(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

export function createScanReportCsv(scans) {
  return [
    reportColumns.map(([label]) => escapeCsv(label)).join(","),
    ...scans.map((scan) => reportColumns.map(([, getValue]) => escapeCsv(getValue(scan))).join(",")),
  ].join("\r\n");
}

export function downloadScanReport(scans, date = new Date()) {
  const report = `\uFEFF${createScanReportCsv(scans)}`;
  const url = URL.createObjectURL(new Blob([report], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `applecare-scan-report-${date.toISOString().slice(0, 10)}.csv`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
