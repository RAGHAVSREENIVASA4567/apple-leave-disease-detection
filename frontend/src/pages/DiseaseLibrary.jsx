import { ArrowUpRight, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import { diseases } from "../data/diseases.js";

export default function DiseaseLibrary() {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => diseases.filter((disease) => `${disease.name} ${disease.type} ${disease.description}`.toLowerCase().includes(query.toLowerCase())), [query]);

  return <div>
    <PageHeader eyebrow="FIELD REFERENCE" title="Apple leaf library" description="A practical starting point for learning about common apple leaf conditions." />
    <div className="library-intro"><span className="library-intro-icon"><Search size={18} /></span><div><strong>Learn the signs, not just the names.</strong><p>Symptoms can overlap and vary by cultivar and climate. Use this guide for context, not as a substitute for a professional diagnosis.</p></div></div>
    <label className="search-field library-search"><Search size={17} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search apple leaf conditions…" aria-label="Search disease library" /></label>
    {filtered.length ? <div className="library-grid">{filtered.map((disease) => <Link to={`/diseases/${disease.slug}`} className="library-card" key={disease.slug}><div className="library-image"><img src={disease.image} alt="" /><span style={{ "--disease-color": disease.color }}>{disease.type}</span></div><div className="library-card-content"><h2>{disease.name}</h2><p>{disease.description}</p><span className="library-link">Explore condition <ArrowUpRight size={15} /></span></div></Link>)}</div> : <div className="empty-state"><strong>No matching conditions</strong><p>Try a different search term.</p></div>}
  </div>;
}
