import { useState } from "react";

const API_BASE = "http://localhost:8000";
const SEV_COLOR = { Low: "#3ecf8e", Medium: "#f5a623", High: "#ff5c5c" };

function FrequencyBadge({ count, total }) {
  const pct  = total > 0 ? (count / total) * 100 : 0;
  const color = pct >= 60 ? "#ff5c5c" : pct >= 30 ? "#f5a623" : "#6b7fa3";
  const label = pct >= 60 ? "Persistent" : pct >= 30 ? "Recurring" : "Occasional";
  return (
    <span style={{
      padding: "2px 8px", borderRadius: 12, fontSize: 10, fontWeight: 700,
      background: `${color}20`, border: `1px solid ${color}40`, color,
    }}>
      {label} ({count}/{total})
    </span>
  );
}

export default function LongitudinalReport() {
  const [patientId, setPatientId] = useState("");
  const [report,    setReport]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const load = async (id) => {
    const pid = id || patientId;
    if (!pid.trim()) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_BASE}/longitudinal-report/${encodeURIComponent(pid)}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Error ${res.status}`);
      setReport(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const trajColor = report?.trajectory === "improving" ? "#3ecf8e"
                  : report?.trajectory === "worsening"  ? "#ff5c5c" : "#f5a623";
  const trajIcon  = report?.trajectory === "improving" ? "📈" 
                  : report?.trajectory === "worsening"  ? "📉" : "📊";

  return (
    <div>
      <h1>Longitudinal Patient Report</h1>
      <p className="page-subtitle">
        Multi-session analysis — symptom evolution, therapy history, and clinical trajectory.
      </p>

      {/* Search */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="flex-row">
          <div style={{ flex: 1 }}>
            <label>Patient ID or Name</label>
            <input type="text" placeholder="e.g. sara_patel" value={patientId}
              onChange={e => setPatientId(e.target.value)}
              onKeyDown={e => e.key === "Enter" && load()} />
          </div>
          <div style={{ paddingTop: 20 }}>
            <button className="btn btn-primary" onClick={() => load()} disabled={loading}>
              {loading ? "Loading…" : "📋 Generate Report"}
            </button>
          </div>
        </div>
        {error && <div className="alert alert-error" style={{ marginTop: 12 }}>✕ {error}</div>}
      </div>

      {loading && <div className="loading-container"><div className="spinner" /></div>}

      {report && !loading && (
        <div className="fade-up">

          {/* Header */}
          <div style={{ background: "linear-gradient(135deg,#0d1525,#111e35)", border: "1px solid var(--border)", borderRadius: 14, padding: "24px 28px", marginBottom: 20 }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: 2, marginBottom: 6, textTransform: "uppercase" }}>Longitudinal Clinical Report</div>
            <div style={{ fontFamily: "var(--font-display)", fontSize: 24, marginBottom: 8 }}>{report.patient?.name}</div>
            <div className="grid-4" style={{ gap: 16, marginTop: 16 }}>
              {[
                { label: "Total Sessions", value: report.total_sessions },
                { label: "First Severity",  value: report.first_severity?.toFixed(2) },
                { label: "Latest Severity", value: report.last_severity?.toFixed(2) },
                { label: "Change",          value: `${report.severity_change > 0 ? "+" : ""}${report.severity_change?.toFixed(2)}` },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "12px 16px" }}>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{label.toUpperCase()}</div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Trajectory */}
          <div className="card" style={{ marginBottom: 20, borderLeft: `3px solid ${trajColor}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 28 }}>{trajIcon}</span>
              <div>
                <div style={{ fontWeight: 700, fontSize: 16, color: trajColor, textTransform: "capitalize" }}>
                  Overall Trajectory: {report.trajectory}
                </div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
                  {report.trajectory === "improving"
                    ? `Severity reduced by ${Math.abs(report.severity_change)?.toFixed(2)} points across ${report.total_sessions} sessions. Treatment appears effective.`
                    : report.trajectory === "worsening"
                    ? `Severity increased by ${Math.abs(report.severity_change)?.toFixed(2)} points. Clinical review and treatment adjustment recommended.`
                    : `Severity has remained stable across ${report.total_sessions} sessions. Continue current treatment plan.`
                  }
                </div>
              </div>
            </div>
          </div>

          {/* Symptom frequency - Feature 11 */}
          <div className="card" style={{ marginBottom: 20 }}>
            <h2>Symptom Frequency Analysis</h2>
            <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
              Tracks how often each symptom appeared across all {report.total_sessions} sessions.
            </p>
            <div className="grid-3" style={{ gap: 16 }}>
              {/* Persistent */}
              <div style={{ background: "rgba(255,92,92,0.06)", border: "1px solid rgba(255,92,92,0.2)", borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#ff5c5c", marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.8 }}>
                  🔴 Persistent (≥60% sessions)
                </div>
                {Object.keys(report.persistent_symptoms || {}).length === 0
                  ? <div style={{ color: "var(--text-muted)", fontSize: 12 }}>None</div>
                  : Object.entries(report.persistent_symptoms).map(([sym, cnt]) => (
                      <div key={sym} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12 }}>
                        <span style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>{sym}</span>
                        <FrequencyBadge count={cnt} total={report.total_sessions} />
                      </div>
                    ))
                }
              </div>
              {/* Recurring */}
              <div style={{ background: "rgba(245,166,35,0.06)", border: "1px solid rgba(245,166,35,0.2)", borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#f5a623", marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.8 }}>
                  🟡 Recurring (30–60%)
                </div>
                {Object.keys(report.recurring_symptoms || {}).length === 0
                  ? <div style={{ color: "var(--text-muted)", fontSize: 12 }}>None</div>
                  : Object.entries(report.recurring_symptoms).map(([sym, cnt]) => (
                      <div key={sym} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12 }}>
                        <span style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>{sym}</span>
                        <FrequencyBadge count={cnt} total={report.total_sessions} />
                      </div>
                    ))
                }
              </div>
              {/* Occasional */}
              <div style={{ background: "rgba(107,127,163,0.06)", border: "1px solid rgba(107,127,163,0.2)", borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#6b7fa3", marginBottom: 10, textTransform: "uppercase", letterSpacing: 0.8 }}>
                  ⚪ Occasional (&lt;30%)
                </div>
                {Object.keys(report.occasional_symptoms || {}).length === 0
                  ? <div style={{ color: "var(--text-muted)", fontSize: 12 }}>None</div>
                  : Object.entries(report.occasional_symptoms).map(([sym, cnt]) => (
                      <div key={sym} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12 }}>
                        <span style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>{sym}</span>
                        <FrequencyBadge count={cnt} total={report.total_sessions} />
                      </div>
                    ))
                }
              </div>
            </div>
          </div>

          {/* Therapies tried + disorders */}
          <div className="grid-2" style={{ gap: 16, marginBottom: 20 }}>
            <div className="card">
              <h2>Therapies Tried</h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {(report.therapies_tried || []).map((t, i) => (
                  <div key={t} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", background: "var(--bg-deep)", borderRadius: 8, border: "1px solid var(--green-soft)" }}>
                    <div style={{ width: 22, height: 22, borderRadius: "50%", background: "var(--green-soft)", color: "var(--green)", fontSize: 11, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center" }}>{i+1}</div>
                    <span style={{ fontSize: 13, color: "var(--green)" }}>{t}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <h2>Diagnoses Across Sessions</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {(report.disorders_seen || []).map(d => (
                  <span key={d} className="tag tag-red">{d}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Session timeline */}
          <div className="card">
            <h2>Session History</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {(report.sessions || []).map((s, i) => {
                const col = SEV_COLOR[s.severity_label] || "var(--text-muted)";
                return (
                  <div key={s.session_id} style={{ display: "flex", gap: 16, padding: "14px 16px", background: "var(--bg-deep)", borderRadius: 10, border: "1px solid var(--border-soft)", alignItems: "flex-start" }}>
                    <div style={{ width: 28, height: 28, borderRadius: "50%", background: `${col}20`, border: `2px solid ${col}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: col, flexShrink: 0 }}>{i+1}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
                        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{(s.date || "").split("T")[0]}</span>
                        <div style={{ display: "flex", gap: 6 }}>
                          <span className="tag tag-red">{s.predicted_disorder}</span>
                          <span style={{ fontSize: 11, fontWeight: 700, color: col, padding: "3px 8px", border: `1px solid ${col}40`, borderRadius: 12 }}>
                            {s.severity_label} {s.severity_score?.toFixed(1)}
                          </span>
                        </div>
                      </div>
                      <div style={{ fontSize: 12, color: "var(--green)", marginBottom: 6 }}>→ {s.recommended_therapy}</div>
                      {s.symptoms?.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {s.symptoms.map(sym => (
                            <span key={sym} style={{ padding: "2px 8px", borderRadius: 10, background: "rgba(245,166,35,0.1)", border: "1px solid rgba(245,166,35,0.2)", color: "#f5a623", fontSize: 10 }}>
                              {sym}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}