export function filterScansByRange(scans, range, now = Date.now()) {
  if (range === "all") return scans;

  const days = Number(range);
  if (!Number.isFinite(days) || days <= 0) return scans;

  const cutoff = now - days * 24 * 60 * 60 * 1000;
  return scans.filter((scan) => {
    const createdAt = new Date(scan.createdAt).getTime();
    return Number.isFinite(createdAt) && createdAt >= cutoff && createdAt <= now;
  });
}
