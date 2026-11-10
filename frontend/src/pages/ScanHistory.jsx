import { useMemo, useState } from "react";
import { ArrowDownToLine, ArrowRight, Search, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { useHistory } from "../hooks/useHistory.js";
import { deleteHistoryItem } from "../services/api.js";
import { downloadScanReport } from "../utils/scanReport.js";

export default function ScanHistory() {
  const { scans, loading, error, refresh, setScans } = useHistory();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [actionError, setActionError] = useState("");

  const filtered = useMemo(() => scans.filter((scan) => {
    const searchText = `${scan.filename} ${scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || ""} ${scan.analysis.health_status}`.toLowerCase();
    return searchText.includes(query.toLowerCase()) && (filter === "all" || scan.analysis.health_status === filter);
  }), [scans, query, filter]);

  async function removeScan(scan) {
    if (!window.confirm(`Delete the scan "${scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || scan.filename}"? This cannot be undone.`)) return;
    try {
      await deleteHistoryItem(scan.id);
      setScans((items) => items.filter((item) => item.id !== scan.id));
      setActionError("");
    } catch (requestError) {
      setActionError(requestError.message);
    }
  }

  function exportScans() {
    downloadScanReport(filtered);
  }

  return <div>
    <PageHeader eyebrow="YOUR ORCHARD RECORD" title="Scan history" description="Review, search, and manage your previous leaf assessments." action={<button className="button button-outline" disabled={!filtered.length} onClick={exportScans}><ArrowDownToLine size={15} /> Export CSV</button>} />
    <section className="panel history-panel">
      <div className="history-toolbar"><label className="search-field"><Search size={17} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search analyses…" aria-label="Search scans" /></label><div className="filter-tabs" aria-label="Filter scans">{[["all", "All scans"], ["healthy", "Healthy"], ["potential_disease", "Potential disease"], ["uncertain", "Uncertain"]].map(([value, label]) => <button key={value} className={filter === value ? "filter-active" : ""} onClick={() => setFilter(value)}>{label}</button>)}</div><span className="scan-count">{filtered.length} {filtered.length === 1 ? "scan" : "scans"}</span></div>
      {(error || actionError) && <div className="notice notice-error">{actionError || error}</div>}
      {loading ? <div className="empty-state compact">Loading scan history…</div> : filtered.length ? <div className="history-table-wrap"><table className="history-table"><thead><tr><th>PHOTO & MODEL CLASS</th><th>HEALTH STATUS</th><th>DATE ADDED</th><th>MODEL</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>{filtered.map((scan) => { const classTitle = scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || "Assessment uncertain"; return <tr key={scan.id}><td><Link to={`/analysis/${scan.id}`} className="history-scan-cell"><img src={scan.imageData} alt="" /><span><strong>{classTitle}</strong><small>{scan.filename}</small></span></Link></td><td><StatusBadge status={scan.analysis.health_status} /></td><td>{new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(scan.createdAt))}</td><td><span className="model-label">{scan.model}</span></td><td className="history-actions"><Link to={`/analysis/${scan.id}`} aria-label="View analysis"><ArrowRight size={16} /></Link><button onClick={() => removeScan(scan)} aria-label={`Delete ${classTitle}`}><Trash2 size={16} /></button></td></tr>; })}</tbody></table></div> : <div className="empty-state"><span className="empty-icon"><Search size={21} /></span><strong>{scans.length ? "No scans match your search" : "No scans saved yet"}</strong><p>{scans.length ? "Try another search or filter." : <>Your analyses will appear here after you scan a leaf.<br /><Link className="button button-dark button-small" to="/scanner">Start a scan</Link></>}</p></div>}
      {!loading && scans.length > 0 && <button className="history-refresh" onClick={refresh}>Refresh history</button>}
    </section>
  </div>;
}
