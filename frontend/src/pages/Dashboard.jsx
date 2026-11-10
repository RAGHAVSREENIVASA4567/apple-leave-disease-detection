import { ArrowRight, ArrowUpRight, ImagePlus, Leaf, ScanLine, Sprout, TrendingUp } from "lucide-react";
import { useMemo, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { useHistory } from "../hooks/useHistory.js";
import { filterScansByRange } from "../utils/dashboardMetrics.js";

const chartColors = ["#35845a", "#db9755", "#a9c5ad", "#7d8f78", "#c5d5c8", "#67786e"];

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
}

export default function Dashboard() {
  const { scans, loading, error } = useHistory();
  const [range, setRange] = useState("all");
  const filteredScans = useMemo(() => filterScansByRange(scans, range), [range, scans]);
  const healthy = filteredScans.filter((scan) => scan.analysis.health_status === "healthy").length;
  const potential = filteredScans.filter((scan) => scan.analysis.health_status === "potential_disease").length;
  const classCounts = filteredScans.reduce((counts, scan) => {
    const classTitle = scan.analysis.predicted_class?.title || scan.analysis.predicted_disease;
    if (classTitle && classTitle !== "Uncertain") counts[classTitle] = (counts[classTitle] || 0) + 1;
    return counts;
  }, {});
  const chartData = Object.entries(classCounts).map(([name, value]) => ({ name, value }));
  const rangeLabel = range === "all" ? "All time" : `Last ${range} days`;

  return (
    <div>
      <PageHeader eyebrow="ORCHARD OVERVIEW" title="Good morning, Jordan" description="Here’s a look at your recent apple leaf assessments." action={<Link className="button button-dark" to="/scanner"><ScanLine size={16} /> New analysis</Link>} />
      {error && <div className="notice notice-error">{error}</div>}
      <section className="stats-grid">
        <StatCard label="Total analyses" value={filteredScans.length} note={range === "all" ? "Saved in scan history" : rangeLabel} icon={ScanLine} tone="green" />
        <StatCard label="Healthy assessments" value={healthy} note={filteredScans.length ? `${Math.round((healthy / filteredScans.length) * 100)}% of this period` : "No assessments yet"} icon={Sprout} tone="soft" />
        <StatCard label="Potential disease" value={potential} note="Worth another look" icon={Leaf} tone="amber" />
        <div className="stat-card stat-card-tip"><div className="tip-glow" /><div className="tip-top"><span className="tip-icon"><TrendingUp size={17} /></span><span>FIELD NOTE</span></div><strong>Small details can tell a story.</strong><p>Take clear photos in natural light and look at more than one leaf.</p><Link to="/scanner">Scan a leaf <ArrowRight size={14} /></Link></div>
      </section>
      <section className="dashboard-panels">
        <div className="panel recent-panel"><div className="panel-head"><div><div className="eyebrow">YOUR LATEST CHECK-INS</div><h2>Recent analyses</h2></div><Link to="/history" className="panel-link">View history <ArrowRight size={15} /></Link></div>
          {loading ? <div className="empty-state compact">Loading your scan history…</div> : filteredScans.length ? <div className="recent-list">{filteredScans.slice(0, 5).map((scan) => <Link className="recent-row" to={`/analysis/${scan.id}`} key={scan.id}><img src={scan.imageData} alt="" /><span className="recent-details"><strong>{scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || "Assessment uncertain"}</strong><small>{formatDate(scan.createdAt)} · {scan.model}</small></span><StatusBadge status={scan.analysis.health_status} /><ArrowUpRight className="recent-arrow" size={16} /></Link>)}</div> : scans.length ? <div className="empty-state compact">No analyses in this time range. Try a wider range.</div> : <div className="empty-state"><span className="empty-icon"><ImagePlus size={22} /></span><strong>Your first scan starts here</strong><p>Upload or take a leaf photo to rank the model’s four classes.</p><Link className="button button-dark button-small" to="/scanner">Scan a leaf <ArrowRight size={14} /></Link></div>}
        </div>
        <div className="panel chart-panel"><div className="panel-head"><div><div className="eyebrow">A QUICK PICTURE</div><h2>Top model classes</h2></div><div className="range-switch" role="group" aria-label="Dashboard date range">{[["7", "7D"], ["30", "30D"], ["all", "All"]].map(([value, label]) => <button key={value} className={range === value ? "range-active" : ""} aria-pressed={range === value} onClick={() => setRange(value)}>{label}</button>)}</div></div>
          {chartData.length ? <><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={chartData} dataKey="value" nameKey="name" innerRadius={66} outerRadius={92} paddingAngle={4} stroke="none">{chartData.map((entry, index) => <Cell key={entry.name} fill={chartColors[index % chartColors.length]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer><div className="chart-center"><strong>{filteredScans.length}</strong><span>ranked<br />images</span></div></div><div className="chart-legend">{chartData.slice(0, 4).map((entry, index) => <div key={entry.name}><i style={{ background: chartColors[index % chartColors.length] }} /><span>{entry.name}</span><strong>{entry.value}</strong></div>)}</div><p className="chart-note">Counts each scan’s highest-ranked class, including uncertain results.</p></> : <div className="chart-empty"><div className="chart-empty-illustration"><Leaf size={23} /></div><strong>Your insights will grow here.</strong><p>Complete a few scans to see the model’s top-class counts across your history.</p></div>}
        </div>
      </section>
      <section className="dashboard-banner"><div className="banner-leaf"><Leaf size={24} /></div><div><strong>A helpful first look, never the final word.</strong><p>The classifier returns one of four class scores and does not explain visible evidence. Confirm results with a local agricultural expert.</p></div><Link to="/diseases">Browse the library <ArrowRight size={15} /></Link></section>
    </div>
  );
}

function StatCard({ label, value, note, icon: Icon, tone }) {
  return <div className={`stat-card stat-${tone}`}><div className="stat-card-top"><span>{label}</span><span className="stat-icon"><Icon size={18} /></span></div><strong className="stat-value">{value}</strong><span className="stat-note">{note}</span></div>;
}
