import { useEffect, useState } from "react";
import { AlertTriangle, ArrowLeft, CalendarDays, CircleHelp, ShieldCheck } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { getHistoryItem } from "../services/api.js";

export default function Results() {
  const { id } = useParams();
  const location = useLocation();
  const [scan, setScan] = useState(location.state?.scan || null);
  const [loading, setLoading] = useState(!location.state?.scan);
  const [error, setError] = useState("");

  useEffect(() => {
    if (scan?.id === id) return;
    let active = true;
    setLoading(true);
    getHistoryItem(id).then((item) => { if (active) setScan(item); })
      .catch((requestError) => { if (active) setError(requestError.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [id, scan?.id]);

  if (loading) return <div className="empty-state result-loading">Loading your assessment…</div>;
  if (error || !scan) return <div className="empty-state result-loading"><span className="empty-icon"><CircleHelp size={22} /></span><strong>Assessment unavailable</strong><p>{error || "This analysis could not be found in your history."}</p><Link className="button button-dark button-small" to="/history">Go to scan history</Link></div>;

  const analysis = scan.analysis;
  const classScores = analysis.class_scores || [];
  const topClass = analysis.predicted_class?.title || analysis.predicted_disease || "Assessment uncertain";
  const topScore = typeof analysis.confidence_score === "number" ? analysis.confidence_score : null;
  const isClassifierResult = Boolean(analysis.predicted_class && classScores.length);
  const displayTitle = analysis.health_status === "uncertain" ? "Assessment uncertain" : topClass;
  const imageQualityLabel = analysis.image_quality === "not_assessed" ? "Not assessed" : analysis.image_quality || "Not assessed";
  const summary = isClassifierResult
    ? `Highest-ranked model class: ${topClass}. The classifier returns class scores only; it does not identify visual symptoms, causes, or treatment.`
    : analysis.is_apple_leaf
      ? analysis.full_analysis
      : "This earlier assessment did not verify an identifiable apple leaf.";

  return <div>
    <PageHeader eyebrow="IMAGE ASSESSMENT" title="Your leaf, in focus." description={isClassifierResult ? "A local four-class MobileNetV2 result. It cannot verify leaf identity or explain visible evidence." : "An earlier saved assessment. Review its limitations and confirm with a local expert."} action={<Link className="button button-outline" to="/history"><ArrowLeft size={15} /> Scan history</Link>} />
    <div className="results-grid">
      <section className="panel result-image-panel"><div className="result-image-wrap"><img src={scan.imageData} alt="Submitted leaf photograph" /></div><div className="result-photo-meta"><span><CalendarDays size={15} />{new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(scan.createdAt))}</span><span>{scan.filename}</span></div><div className="image-quality-row"><span>Image quality</span><strong className="quality-not_assessed">{imageQualityLabel}</strong></div></section>
      <section className="panel diagnosis-panel"><div className="diagnosis-topline"><span className="eyebrow">{isClassifierResult ? "MOBILENETV2 CLASSIFIER" : "SAVED HISTORICAL ASSESSMENT"}</span><StatusBadge status={analysis.health_status} /></div><h2>{displayTitle}</h2><p className="diagnosis-summary">{summary}</p>
        <div className="confidence-row"><span>{topScore === null ? "Model-reported confidence" : "Top-class score"}</span><span className="confidence-pill">{topScore === null ? analysis.confidence || "Unavailable" : `${(topScore * 100).toFixed(1)}%`}</span><small>{isClassifierResult ? "Uncalibrated model score—not diagnostic confidence" : "Qualitative estimate from the earlier assessment"}</small></div>
        {analysis.uncertainty_notes && <div className="uncertainty-box"><AlertTriangle size={17} /><div><strong>Important limitations</strong><p>{analysis.uncertainty_notes}</p></div></div>}
        <div className="assessment-meta"><span>{scan.provider} · {scan.model}</span><span><ShieldCheck size={14} /> {isClassifierResult ? "Local model · screening only" : "Historical result · visual guidance only"}</span></div>
      </section>
    </div>
    {isClassifierResult && <section className="panel class-scores-panel"><div className="panel-head"><div><div className="eyebrow">MODEL OUTPUT</div><h2>Scores for all four classes</h2></div></div><p className="class-scores-note">These are the model’s raw softmax scores. They are not calibrated probabilities and do not indicate that a specific symptom was observed.</p><div className="class-score-list">{classScores.map((item) => <div className="class-score-row" key={item.label}><span>{item.title}</span><div className="class-score-track"><span style={{ width: `${Math.min(100, Math.max(0, item.score * 100))}%` }} /></div><strong>{(item.score * 100).toFixed(1)}%</strong></div>)}</div></section>}
    <div className="expert-note"><span><ShieldCheck size={19} /></span><div><strong>A helpful first look—not a confirmed diagnosis.</strong><p>{isClassifierResult ? "The model always ranks one of its four training classes, even for an unfamiliar image. Confirm results with a local agricultural extension service or plant pathologist before choosing a treatment." : "Earlier visual assessments can be wrong, and symptoms can overlap. Confirm results with a local agricultural extension service or plant pathologist before choosing a treatment."}</p></div><Link to="/diseases">Explore the library</Link></div>
  </div>;
}
