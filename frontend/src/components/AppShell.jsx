import { Activity, BookOpen, ChevronDown, CircleHelp, LayoutDashboard, Leaf, Menu, ScanLine, Settings2, X } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import Brand from "./Brand.jsx";
import { getApiHealth } from "../services/api.js";

const links = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { to: "/scanner", label: "AI Leaf Scanner", icon: ScanLine },
  { to: "/history", label: "Scan history", icon: Activity },
  { to: "/diseases", label: "Disease library", icon: BookOpen },
];

export default function AppShell() {
  const [open, setOpen] = useState(false);
  const [serviceStatus, setServiceStatus] = useState("checking");
  const location = useLocation();

  useEffect(() => {
    let active = true;
    const checkModel = () => getApiHealth()
      .then((health) => { if (active) setServiceStatus(health.modelReady ? "configured" : "setup-needed"); })
      .catch(() => { if (active) setServiceStatus("unavailable"); });
    checkModel();
    const timer = window.setInterval(checkModel, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  return (
    <div className="app-shell">
      {open && <button className="mobile-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="sidebar-brand">
          <Brand />
          <button className="icon-button mobile-close" onClick={() => setOpen(false)} aria-label="Close menu"><X size={19} /></button>
        </div>
        <div className="workspace-select">
          <span className="workspace-avatar">O</span>
          <span className="workspace-copy"><strong>Orchard workspace</strong><small>Grower plan</small></span>
          <ChevronDown size={16} />
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="sidebar-nav">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
              {label === "AI Leaf Scanner" && <span className="nav-new">AI</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-help"><span className="help-icon"><CircleHelp size={17} /></span><div><strong>Need a hand?</strong><small>Explore our guide</small></div><ChevronDown size={15} /></div>
          <div className="profile-row"><span className="profile-avatar">JD</span><span className="profile-copy"><strong>Jordan Davis</strong><small>Orchard manager</small></span><Settings2 size={17} /></div>
        </div>
      </aside>
      <div className="shell-main">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={21} /></button>
          <div className="breadcrumb"><span>AppleCare AI</span><span className="crumb-slash">/</span><strong>{links.find((link) => location.pathname.startsWith(link.to))?.label || "Analysis"}</strong></div>
          <div className="topbar-right"><span className={`status-dot service-${serviceStatus}`} /> <span>{serviceStatus === "checking" ? "Checking model" : serviceStatus === "configured" ? "Local model ready" : serviceStatus === "setup-needed" ? "Model service needed" : "API unavailable"}</span><span className="topbar-divider" /><span className="avatar-small">JD</span></div>
        </header>
        <main className="main-content"><Outlet /></main>
      </div>
    </div>
  );
}
