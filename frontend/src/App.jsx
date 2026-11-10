import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell.jsx";

const Dashboard = lazy(() => import("./pages/Dashboard.jsx"));
const DiseaseDetail = lazy(() => import("./pages/DiseaseDetail.jsx"));
const DiseaseLibrary = lazy(() => import("./pages/DiseaseLibrary.jsx"));
const Home = lazy(() => import("./pages/Home.jsx"));
const Results = lazy(() => import("./pages/Results.jsx"));
const Scanner = lazy(() => import("./pages/Scanner.jsx"));
const ScanHistory = lazy(() => import("./pages/ScanHistory.jsx"));

export default function App() {
  return (
    <Suspense fallback={<div className="app-loading">Loading AppleCare AI…</div>}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route element={<AppShell />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/history" element={<ScanHistory />} />
          <Route path="/diseases" element={<DiseaseLibrary />} />
          <Route path="/diseases/:slug" element={<DiseaseDetail />} />
          <Route path="/analysis/:id" element={<Results />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
