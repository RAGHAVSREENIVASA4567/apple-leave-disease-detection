import { ArrowLeft, ArrowRight, CircleHelp, Leaf, ShieldCheck, Sprout } from "lucide-react";
import { Link, Navigate, useParams } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import { diseases } from "../data/diseases.js";

export default function DiseaseDetail() {
  const { slug } = useParams();
  const disease = diseases.find((item) => item.slug === slug);
  if (!disease) return <Navigate to="/diseases" replace />;

  return <div>
    <PageHeader eyebrow={disease.type.toUpperCase()} title={disease.name} description={disease.description} action={<Link className="button button-outline" to="/diseases"><ArrowLeft size={15} /> All conditions</Link>} />
    <div className="disease-detail-grid"><div className="disease-detail-image"><img src={disease.image} alt="" /><span>Illustrative photo · not a symptom reference</span></div><div className="disease-detail-aside"><div className="detail-aside-icon"><CircleHelp size={19} /></div><h3>Keep the whole tree in view.</h3><p>One leaf photo cannot confirm this condition. Check multiple leaves, the fruit and nearby plant context, and note recent weather.</p><Link className="button button-dark button-small" to="/scanner">Scan a leaf <ArrowRight size={14} /></Link></div></div>
    <div className="detail-info-grid"><InfoBlock title="Visible signs to look for" icon={Leaf} items={disease.symptoms} /><InfoBlock title="Possible contributing factors" icon={CircleHelp} items={disease.causes} /><InfoBlock title="Prevention & monitoring" icon={Sprout} items={disease.prevention} /></div>
    <div className="expert-note"><span><ShieldCheck size={19} /></span><div><strong>Diagnosis and treatment depend on local conditions.</strong><p>Contact your regional agricultural extension service or a plant pathologist for confirmation and locally appropriate recommendations.</p></div></div>
  </div>;
}

function InfoBlock({ title, icon: Icon, items }) {
  return <section className="panel detail-info-card"><div className="result-card-heading"><span><Icon size={17} /></span><h3>{title}</h3></div><ul>{items.map((item) => <li key={item}>{item}</li>)}</ul></section>;
}
