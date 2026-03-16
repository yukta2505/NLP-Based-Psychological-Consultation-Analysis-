import { useState } from "react";
import Dashboard from "./components/Dashboard";
import AnalysisPage from "./pages/AnalysisPage";
import PatientHistoryPage from "./pages/PatientHistoryPage";
import MetricsPage from "./pages/MetricsPage";
import LongitudinalReport from "./pages/Longitudinalreport";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [analysisResult, setAnalysisResult] = useState(null);

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="logo">
          <span className="logo-icon">🧠</span>
          <span className="logo-text">PsychNLP</span>
        </div>
        <ul>
          {[
            { id: "dashboard",  icon: "⬡", label: "Dashboard" },
            { id: "analyze",    icon: "◈", label: "Analyze" },
            { id: "history",    icon: "◉", label: "Patient History" },
            { id: "metrics",     icon: "◎", label: "Model Metrics" },
            { id: "longitudinal", icon: "📋", label: "Case Report" },
          ].map(({ id, icon, label }) => (
            <li
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => setPage(id)}
            >
              <span className="nav-icon">{icon}</span>
              <span>{label}</span>
            </li>
          ))}
        </ul>
        <div className="sidebar-footer">
          <span>NLP Research System v1.0</span>
        </div>
      </nav>

      <main className="main-content">
        {page === "dashboard"  && <Dashboard setPage={setPage} />}
        {page === "analyze"    && (
          <AnalysisPage
            setResult={setAnalysisResult}
            result={analysisResult}
          />
        )}
        {page === "history"    && <PatientHistoryPage />}
        {page === "metrics"    && <MetricsPage />}
        {page === "longitudinal" && <LongitudinalReport />}
      </main>
    </div>
  );
}