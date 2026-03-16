import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const SEV_COLOR = { Low: "var(--green)", Medium: "var(--yellow)", High: "var(--red)" };
const SEV_NUM   = { Low: 1, Medium: 2, High: 3 };

// ── Session Comparison Timeline ──────────────────────────────────
function SessionTimeline({ sessions }) {
  if (!sessions || sessions.length < 2) return null;

  const reversed = [...sessions].reverse(); // oldest first

  return (
    <div style={{ marginBottom: 24 }}>
      {/* Timeline track */}
      <div style={{ position: "relative", padding: "0 20px" }}>

        {/* Horizontal line */}
        <div style={{
          position: "absolute", top: 36, left: 40,
          right: 40, height: 2,
          background: "linear-gradient(90deg, var(--accent), var(--green))",
          opacity: 0.3,
        }} />

        {/* Session dots */}
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 32 }}>
          {reversed.map((s, i) => {
            const col = SEV_COLOR[s.severity_label] || "var(--accent)";
            return (
              <div key={s.session_id} style={{ textAlign: "center", flex: 1 }}>
                <div style={{
                  width: 14, height: 14, borderRadius: "50%",
                  background: col, margin: "29px auto 8px",
                  boxShadow: `0 0 8px ${col}80`,
                  border: "2px solid var(--bg-deep)",
                  position: "relative", zIndex: 1,
                }} />
                <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                  {(s.date || "").split("T")[0]}
                </div>
                <div style={{ fontSize: 10, fontWeight: 700, color: col }}>
                  S{i + 1}
                </div>
              </div>
            );
          })}
        </div>

        {/* Comparison cards */}
        <div style={{
          display: "grid",
          gridTemplateColumns: `repeat(${Math.min(reversed.length, 4)}, 1fr)`,
          gap: 10,
        }}>
          {reversed.map((s, i) => {
            const col     = SEV_COLOR[s.severity_label] || "var(--accent)";
            const prevSev = i > 0 ? SEV_NUM[reversed[i-1].severity_label] || 2 : null;
            const curSev  = SEV_NUM[s.severity_label] || 2;
            const trend   = prevSev === null ? null
              : curSev < prevSev ? "↓ Improving"
              : curSev > prevSev ? "↑ Worsening"
              : "→ Stable";
            const trendCol = trend?.includes("Improving") ? "var(--green)"
              : trend?.includes("Worsening") ? "var(--red)" : "var(--yellow)";

            return (
              <div key={s.session_id} style={{
                background: "var(--bg-card)",
                border: `1px solid ${col}40`,
                borderTop: `3px solid ${col}`,
                borderRadius: 10,
                padding: "12px 14px",
              }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                  Session {i + 1}
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: col, marginBottom: 6 }}>
                  {s.severity_label}
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", marginLeft: 4 }}>
                    {s.severity_score?.toFixed(1)}
                  </span>
                </div>
                {trend && (
                  <div style={{ fontSize: 10, color: trendCol, fontWeight: 600, marginBottom: 6 }}>
                    {trend}
                  </div>
                )}
                <div style={{
                  fontSize: 11, color: "var(--text-secondary)",
                  padding: "4px 8px",
                  background: "var(--bg-deep)",
                  borderRadius: 6, marginBottom: 4,
                }}>
                  {s.predicted_disorder}
                </div>
                <div style={{
                  fontSize: 10, color: "var(--green)",
                  padding: "3px 8px",
                  background: "var(--green-soft)",
                  borderRadius: 6,
                }}>
                  {s.recommended_therapy}
                </div>
                {Array.isArray(s.symptoms) && s.symptoms.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 3 }}>
                      SYMPTOMS
                    </div>
                    {s.symptoms.slice(0, 3).map(sym => (
                      <span key={sym} style={{
                        display: "inline-block", fontSize: 9,
                        background: "rgba(245,166,35,0.1)",
                        color: "var(--yellow)",
                        border: "1px solid rgba(245,166,35,0.2)",
                        borderRadius: 10, padding: "1px 6px",
                        marginRight: 3, marginBottom: 3,
                      }}>
                        {sym}
                      </span>
                    ))}
                    {s.symptoms.length > 3 && (
                      <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
                        +{s.symptoms.length - 3} more
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Summary insight */}
        {reversed.length >= 2 && (() => {
          const first = SEV_NUM[reversed[0].severity_label] || 2;
          const last  = SEV_NUM[reversed[reversed.length - 1].severity_label] || 2;
          const diff  = first - last;
          const msg   = diff > 0
            ? `Overall improvement observed across ${reversed.length} sessions. Severity reduced from ${reversed[0].severity_label} to ${reversed[reversed.length-1].severity_label}.`
            : diff < 0
            ? `Severity has increased across sessions. Clinical review recommended.`
            : `Severity has remained stable across ${reversed.length} sessions.`;
          const msgCol = diff > 0 ? "var(--green)" : diff < 0 ? "var(--red)" : "var(--yellow)";
          return (
            <div style={{
              marginTop: 14, padding: "10px 16px",
              background: "var(--bg-card)",
              border: `1px solid ${msgCol}30`,
              borderLeft: `3px solid ${msgCol}`,
              borderRadius: 8, fontSize: 13,
              color: "var(--text-secondary)", lineHeight: 1.6,
            }}>
              <span style={{ fontWeight: 600, color: msgCol }}>
                {diff > 0 ? "📈 Progress" : diff < 0 ? "⚠ Review" : "📊 Stable"}:
              </span>{" "}{msg}
            </div>
          );
        })()}
      </div>
    </div>
  );
}


function SeverityChart({ sessions }) {
  if (!sessions || sessions.length < 2) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: 13, textAlign: "center", padding: 20 }}>
        Need at least 2 sessions to show progress chart.
      </div>
    );
  }

  const W = 600, H = 150;
  const pad = { l: 40, r: 20, t: 20, b: 30 };
  const cW = W - pad.l - pad.r;
  const cH = H - pad.t - pad.b;
  const max = 3;

  const pts = sessions.slice().reverse().map((s, i) => ({
    x: pad.l + (i / (sessions.length - 1)) * cW,
    y: pad.t + cH - ((s.severity_score || 0) / max) * cH,
    score: s.severity_score,
    label: s.severity_label,
    date:  (s.date || "").split("T")[0],
  }));

  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaD = `${pathD} L ${pts[pts.length-1].x} ${pad.t+cH} L ${pts[0].x} ${pad.t+cH} Z`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.3"/>
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {[0,1,2,3].map(v => {
        const y = pad.t + cH - (v/max)*cH;
        return (
          <g key={v}>
            <line x1={pad.l} y1={y} x2={W-pad.r} y2={y} stroke="#1d2535" strokeWidth="1"/>
            <text x={pad.l-6} y={y+4} textAnchor="end" fill="#4a5572" fontSize="10">{v}</text>
          </g>
        );
      })}
      <path d={areaD} fill="url(#cg)"/>
      <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="5" fill="var(--accent)" stroke="#0a0d12" strokeWidth="2"/>
          <text x={p.x} y={pad.t+cH+18} textAnchor="middle" fill="#4a5572" fontSize="9">{p.date || `S${i+1}`}</text>
        </g>
      ))}
    </svg>
  );
}

export default function PatientHistoryPage() {
  const [patients,   setPatients]   = useState([]);
  const [patientId,  setPatientId]  = useState("");
  const [history,    setHistory]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [loadingList,setLoadingList]= useState(true);
  const [error,      setError]      = useState(null);

  // Auto-load patient list on mount
  useEffect(() => {
    fetch(`${API_BASE}/all-patients`)
      .then(r => r.json())
      .then(d => setPatients(d.patients || []))
      .catch(() => setPatients([]))
      .finally(() => setLoadingList(false));
  }, []);

  const fetchHistory = async (id) => {
    const pid = id || patientId;
    if (!pid.trim()) return;
    setLoading(true); setError(null); setHistory(null);
    try {
      const res = await fetch(`${API_BASE}/patient-history/${encodeURIComponent(pid)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail;
        if (typeof detail === "object") {
          throw new Error(detail.message + (detail.hint ? " " + detail.hint : ""));
        }
        throw new Error(detail || `Error ${res.status}`);
      }
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPatient = (pid) => {
    setPatientId(pid);
    fetchHistory(pid);
  };

  return (
    <div>
      <h1>Patient History</h1>
      <p className="page-subtitle">
        Track session history and severity trends across consultations.
      </p>

      <div className="grid-2" style={{ gap: 20, marginBottom: 28 }}>

        {/* ── Saved patients list ── */}
        <div className="card">
          <h2>Saved Patients</h2>
          {loadingList ? (
            <div className="loading-container" style={{ padding: 20 }}>
              <div className="spinner" style={{ width: 24, height: 24 }} />
            </div>
          ) : patients.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
              No patients saved yet. Analyze a consultation and click
              <strong> Save Session</strong> to save a patient.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {patients.map(p => (
                <div
                  key={p.patient_id}
                  onClick={() => handleSelectPatient(p.patient_id)}
                  style={{
                    padding: "12px 16px",
                    background: patientId === p.patient_id
                      ? "var(--accent-soft)" : "var(--bg-deep)",
                    border: `1px solid ${patientId === p.patient_id
                      ? "var(--accent)" : "var(--border-soft)"}`,
                    borderRadius: "var(--radius)",
                    cursor: "pointer",
                    transition: "all 0.15s",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 500, fontSize: 14 }}>{p.name}</div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {p.patient_id}
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span className="tag tag-blue">
                      {p.session_count || 0} session{(p.session_count || 0) !== 1 ? "s" : ""}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Manual search ── */}
        <div className="card">
          <h2>Search by ID or Name</h2>
          <label>Patient ID or Name</label>
          <input
            type="text"
            placeholder="e.g. jane_doe or Jane Doe"
            value={patientId}
            onChange={e => setPatientId(e.target.value)}
            onKeyDown={e => e.key === "Enter" && fetchHistory()}
          />
          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-primary"
              onClick={() => fetchHistory()}
              disabled={loading || !patientId.trim()}
              style={{ width: "100%" }}
            >
              {loading ? "Loading…" : "◉  Load History"}
            </button>
          </div>

          {error && (
            <div className="alert alert-error" style={{ marginTop: 12 }}>
              ✕ {error}
            </div>
          )}

          {/* How it works */}
          <div style={{
            marginTop: 16, padding: 12,
            background: "var(--bg-deep)",
            borderRadius: "var(--radius)",
            border: "1px solid var(--border-soft)",
            fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7,
          }}>
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4 }}>
              How session saving works:
            </div>
            1. Go to <strong>Analyze</strong> tab<br/>
            2. Enter patient name and analyze text<br/>
            3. Click <strong>💾 Save Session</strong><br/>
            4. Patient appears in the list on the left<br/>
            5. Click their name to view history
          </div>
        </div>
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner"/>
        </div>
      )}

      {history && !loading && (
        <div className="fade-up">
          {/* Stats */}
          <div className="grid-4" style={{ marginBottom: 24 }}>
            {[
              { label: "Patient",   value: history.patient?.name },
              { label: "Patient ID",value: history.patient?.patient_id },
              { label: "Sessions",  value: history.session_count },
              { label: "Latest",    value: history.sessions?.[0]?.predicted_disorder || "—" },
            ].map(({ label, value }) => (
              <div key={label} className="stat-card">
                <div className="stat-label">{label}</div>
                <div className="stat-value" style={{ fontSize: 16 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Progress chart */}
          {history.sessions?.length >= 2 && (
            <div className="card" style={{ marginBottom: 24 }}>
              <h2>Severity Progress</h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 12, marginBottom: 16 }}>
                Severity score over sessions (0 = healthy, 3 = severe). Lower is better.
              </p>
              <SeverityChart sessions={history.sessions} />
            </div>
          )}

          {/* Session Timeline Comparison */}
          {history.sessions?.length >= 2 && (
            <div className="card" style={{ marginBottom: 24 }}>
              <h2>Session Comparison</h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 20 }}>
                Side-by-side view of all sessions — track disorder, severity, therapy and symptom changes over time.
              </p>
              <SessionTimeline sessions={history.sessions} />
            </div>
          )}

          {/* Sessions */}
          <div className="card">
            <div className="section-header">
              <h2 style={{ margin: 0 }}>All Sessions ({history.session_count})</h2>
            </div>

            {history.sessions?.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No sessions recorded.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {history.sessions.map((s, idx) => (
                  <div key={s.session_id} style={{
                    padding: 16,
                    background: "var(--bg-deep)",
                    borderRadius: "var(--radius)",
                    border: "1px solid var(--border-soft)",
                  }}>
                    {/* Header row */}
                    <div style={{
                      display: "flex", justifyContent: "space-between",
                      alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8,
                    }}>
                      <div>
                        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>
                          Session {history.session_count - idx}
                        </span>
                        <span style={{
                          fontSize: 11, color: "var(--text-muted)",
                          fontFamily: "var(--font-mono)", marginLeft: 10,
                        }}>
                          {(s.date || "").split("T")[0]}
                        </span>
                      </div>
                      <span style={{
                        fontSize: 12, fontWeight: 600,
                        color: SEV_COLOR[s.severity_label],
                        padding: "3px 10px",
                        border: `1px solid ${SEV_COLOR[s.severity_label]}40`,
                        borderRadius: 20,
                      }}>
                        {s.severity_label} — {s.severity_score?.toFixed(2)}
                      </span>
                    </div>

                    {/* Details grid */}
                    <div className="grid-2" style={{ gap: 12, marginBottom: 10 }}>
                      <div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                          DISORDER
                        </div>
                        <span className="tag tag-red">{s.predicted_disorder}</span>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                          THERAPY
                        </div>
                        <span className="tag tag-green">{s.recommended_therapy}</span>
                      </div>
                    </div>

                    {/* Symptoms */}
                    {Array.isArray(s.symptoms) && s.symptoms.length > 0 && (
                      <div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                          SYMPTOMS
                        </div>
                        <div className="entity-list">
                          {s.symptoms.map(sym => (
                            <span key={sym} className="tag tag-yellow">{sym}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Session ID */}
                    <div style={{
                      marginTop: 10, paddingTop: 8,
                      borderTop: "1px solid var(--border-soft)",
                      fontSize: 11, color: "var(--text-muted)",
                      fontFamily: "var(--font-mono)",
                    }}>
                      {s.session_id}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}