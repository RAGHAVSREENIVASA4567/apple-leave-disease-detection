import { ArrowDownRight, ArrowRight, ArrowUpRight, Check, CircleHelp, Leaf, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import Brand from "../components/Brand.jsx";
import { diseases } from "../data/diseases.js";

const orchardImage = "/images/apples-on-tree.webp";
const leafImage = "/images/apple-leaf.webp";

export default function Home() {
  return (
    <div className="landing-page">
      <header className="landing-nav wrap">
        <Brand />
        <nav><a href="#how-it-works">How it works</a><a href="#diseases">Disease library</a><a href="#why-applecare">Why AppleCare</a></nav>
        <div className="landing-actions"><Link className="nav-login" to="/dashboard">Sign in</Link><Link className="button button-dark button-small" to="/scanner">Start scanning <ArrowRight size={15} /></Link></div>
      </header>
      <main>
        <section className="hero-section">
          <div className="hero-image" style={{ backgroundImage: `linear-gradient(90deg, rgba(14,44,31,.91) 0%, rgba(14,44,31,.7) 44%, rgba(14,44,31,.12) 100%), url("${orchardImage}")` }} />
          <div className="hero-content wrap">
            <div className="hero-copy">
              <div className="hero-kicker"><span className="kicker-spark"><Sparkles size={13} /></span> PLANT HEALTH, MADE CLEARER</div>
              <h1>Better leaves.<br />Better <em>harvests.</em></h1>
              <p>Get a closer look at your apple trees. A local MobileNetV2 classifier ranks a leaf photo among four training labels; it cannot verify leaf identity or explain visible symptoms.</p>
              <div className="hero-cta"><Link className="button button-lime" to="/scanner">Scan a leaf <ArrowUpRight size={16} /></Link><a className="hero-secondary" href="#how-it-works">See how it works <ArrowDownRight size={16} /></a></div>
              <div className="hero-trust"><div className="trust-avatars"><span>G</span><span>M</span><span>A</span></div><div><strong>Built for thoughtful growers</strong><small>AI-assisted. Grower-guided.</small></div></div>
            </div>
            <div className="hero-float-card"><span className="float-icon"><ScanLine size={17} /></span><span><strong>One photo, four class scores</strong><small>A limited screening starting point.</small></span><span className="float-live"><i /> READY</span></div>
          </div>
          <div className="hero-index">01 <span /> 03</div>
        </section>
        <section className="intro-strip wrap">
          <span className="intro-label">A clearer first look</span>
          <p>Good decisions start with good observations. AppleCare AI turns a leaf photo into four model class scores—a limited, cautious starting point for your orchard.</p>
          <span className="intro-seal"><ShieldCheck size={18} /> Visual guidance, not a lab diagnosis</span>
        </section>
        <section className="how-section section-pad wrap" id="how-it-works">
          <div className="section-heading-row"><div><div className="eyebrow">A SIMPLE FIRST STEP</div><h2>From leaf to next step.</h2></div><p>Designed to make plant-health observations a little easier, wherever your orchard is.</p></div>
          <div className="steps-grid">
            <article className="step-card"><div className="step-number">01 <span><ScanLine size={20} /></span></div><h3>Capture a clear photo</h3><p>Upload an image or take one in the orchard. A close, well-lit view is a useful starting point for screening.</p><span className="step-line" /></article>
            <article className="step-card"><div className="step-number">02 <span><Sparkles size={20} /></span></div><h3>Review four class scores</h3><p>The local MobileNetV2 model ranks the image against its four training labels. Scores are not calibrated and can be wrong.</p><span className="step-line" /></article>
            <article className="step-card"><div className="step-number">03 <span><Leaf size={20} /></span></div><h3>Check with a local expert</h3><p>Use the separate disease library for general context and confirm any concerns with a local agricultural expert.</p><span className="step-line" /></article>
          </div>
        </section>
        <section className="feature-band" id="why-applecare">
          <div className="feature-inner wrap"><div className="feature-photo" style={{ backgroundImage: `url("${leafImage}")` }}><div className="photo-tag"><Leaf size={15} /> GROW WITH CLARITY</div><div className="photo-caption"><span>01 / 04</span><strong>Healthy orchards begin<br />with paying attention.</strong></div></div><div className="feature-copy">                    <div className="eyebrow eyebrow-pale">MADE FOR THE MOMENTS THAT MATTER</div><h2>A helpful first look.<br />With uncertainty in view.</h2><p>AppleCare AI is a practical companion for a first look. It returns class scores only, clearly notes its limits, and keeps your scan history available.</p><ul><li><Check size={16} /> Four MobileNetV2 output classes</li><li><Check size={16} /> Clear notes when the model is uncertain</li><li><Check size={16} /> Your scan history, ready when you need it</li></ul><Link className="feature-link" to="/dashboard">Explore your dashboard <ArrowRight size={16} /></Link></div></div>
        </section>
        <section className="diseases-section section-pad wrap" id="diseases">
          <div className="section-heading-row"><div><div className="eyebrow">KNOW WHAT TO LOOK FOR</div><h2>Explore common concerns.</h2></div><Link className="text-link" to="/diseases">View disease library <ArrowRight size={16} /></Link></div>
          <div className="disease-preview-grid">{diseases.slice(0, 3).map((disease) => <Link className="disease-preview" to={`/diseases/${disease.slug}`} key={disease.slug}><img src={disease.image} alt="" /><span className="disease-preview-content"><small>{disease.type}</small><strong>{disease.name}</strong><ArrowUpRight size={16} /></span></Link>)}</div>
          <div className="note-inline"><CircleHelp size={15} /> A photo-based assessment cannot confirm a diagnosis. Symptoms can overlap and vary by cultivar and conditions.</div>
        </section>
      </main>
      <footer className="landing-footer"><div className="wrap footer-inner"><Brand light /><span>© 2026 AppleCare AI · For informed growing, one leaf at a time.</span><Link to="/scanner">Start a scan <ArrowRight size={14} /></Link><small className="photo-credit">Photos: <a href="https://commons.wikimedia.org/wiki/File:Fresh_ripe_apples_on_a_tree_(54015357813).jpg" target="_blank" rel="noreferrer">AS Photography</a> (CC BY 4.0) and <a href="https://commons.wikimedia.org/wiki/File:Apple_Tree_Leaf_(35128940).jpeg" target="_blank" rel="noreferrer">Konstantin Tsabolov</a> (CC BY 3.0), resized.</small></div></footer>
    </div>
  );
}
