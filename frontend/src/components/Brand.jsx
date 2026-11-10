import { Leaf } from "lucide-react";
import { Link } from "react-router-dom";

export default function Brand({ light = false }) {
  return (
    <Link to="/" className={`brand ${light ? "brand-light" : ""}`}>
      <span className="brand-icon"><Leaf size={19} strokeWidth={2.2} /></span>
      <span>applecare<span className="brand-ai">.ai</span></span>
    </Link>
  );
}
