import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

const SEV_COLOR = { Low: "#3ecf8e", Medium: "#f5a623", High: "#ff5c5c" };

function MiniBar({ value, max, color }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div style={{ height: 6, background: "#0d1525", borderRadius: 4, overflow: "hidden" }}>
      <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 4, transition: "width 0.6s ease" }} />
    </div>
  );
}

function TrendChart({ data }) {
  if (!data || data.length < 2) return (
    <div style={{ color: "#2a3a55", fontSize: 12, textAlign: "center", padding: "20px 0" }}>
      Save sessions to see the severity trend
    </div>
  );
  const W = 400, H = 80;
  const pad = { l: 30, r: 10, t: 10, b: 20 };
  const cW = W - pad.l - pad.r;
  const cH = H - pad.t - pad.b;
  const pts = data.map((d, i) => ({
    x: pad.l + (i / (data.length - 1)) * cW,
    y: pad.t + cH - ((d.severity_score || 0) / 3) * cH,
    label: d.severity_label,
    score: d.severity_score,
  }));
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const area = `${path} L${pts[pts.length-1].x},${pad.t+cH} L${pts[0].x},${pad.t+cH} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4f8eff" stopOpacity="0.3"/>
          <stop offset="100%" stopColor="#4f8eff" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {[0,1,2,3].map(v => {
        const y = pad.t + cH - (v/3)*cH;
        return <line key={v} x1={pad.l} y1={y} x2={W-pad.r} y2={y} stroke="#0d1525" strokeWidth="1"/>;
      })}
      <path d={area} fill="url(#tg)"/>
      <path d={path} fill="none" stroke="#4f8eff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3.5"
          fill={SEV_COLOR[p.label] || "#4f8eff"}
          stroke="#07090e" strokeWidth="1.5"/>
      ))}
    </svg>
  );
}

export default function Dashboard({ setPage }) {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/dashboard-stats`)
      .then(r => r.json())
      .then(d => setStats(d))
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, []);

  const features = [
    { icon: "◈", title: "Consultation Analysis", desc: "Analyse text or PDF consultation notes with NLP.", color: "var(--accent)", action: "analyze" },
    { icon: "◉", title: "Patient History",       desc: "Track session progress and compare across visits.", color: "var(--green)",  action: "history" },
    { icon: "◎", title: "Model Metrics",         desc: "View F1, accuracy and precision of all models.",  color: "var(--purple)", action: "metrics" },
  ];

  const pipeline = [
    { step:"01", label:"Text Extraction",          desc:"PDF or plain text" },
    { step:"02", label:"Preprocessing",            desc:"Tokenize, lemmatize, clean" },
    { step:"03", label:"Emotion Detection",        desc:"GoEmotions DistilBERT" },
    { step:"04", label:"Disorder Classification",  desc:"DistilBERT fine-tuned" },
    { step:"05", label:"NER Extraction",           desc:"spaCy on NCBI corpus" },
    { step:"06", label:"Therapy Recommendation",   desc:"XGBoost classifier" },
    { step:"07", label:"Risk Detection",           desc:"Clinical keyword flags" },
    { step:"08", label:"Report Generation",        desc:"Patient-friendly PDF" },
  ];

  return (
    <div>
      <h1>Psychological Consultation NLP</h1>
      <p className="page-subtitle">
        Research-grade NLP system for mental health analysis, disorder classification, therapy recommendation and mind map generation.
      </p>

      {/* ── Live stats ── */}
      {loading ? (
        <div className="grid-4" style={{ marginBottom: 28 }}>
          {[1,2,3,4].map(i => (
            <div key={i} className="stat-card" style={{ opacity: 0.4 }}>
              <div className="stat-label">Loading…</div>
              <div className="stat-value" style={{ fontSize: 22 }}>—</div>
            </div>
          ))}
        </div>
      ) : stats ? (
        <div className="grid-4" style={{ marginBottom: 28 }}>
          <div className="stat-card">
            <div className="stat-label">Total Patients</div>
            <div className="stat-value">{stats.total_patients}</div>
            <div className="stat-sub">registered</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Sessions</div>
            <div className="stat-value">{stats.total_sessions}</div>
            <div className="stat-sub">analysed</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Top Disorder (month)</div>
            <div className="stat-value" style={{ fontSize: 15, color: "var(--red)" }}>
              {stats.top_disorder?.predicted_disorder || "N/A"}
            </div>
            <div className="stat-sub">{stats.top_disorder?.cnt || 0} cases</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Avg Severity (month)</div>
            <div className="stat-value" style={{
              fontSize: 22,
              color: stats.avg_severity >= 2 ? "var(--red)" : stats.avg_severity >= 1 ? "var(--yellow)" : "var(--green)"
            }}>
              {stats.avg_severity?.toFixed(2)}
            </div>
            <div className="stat-sub">out of 3.0</div>
          </div>
        </div>
      ) : (
        <div className="alert alert-info" style={{ marginBottom: 24 }}>
          ℹ Backend not connected — start the API to see live statistics.
        </div>
      )}

      {/* Feature cards */}
      <div className="grid-3" style={{ marginBottom: 28 }}>
        {features.map(f => (
          <div key={f.action} className="card" style={{ cursor: "pointer", borderLeft: `3px solid ${f.color}` }} onClick={() => setPage(f.action)}>
            <div style={{ fontSize: 24, marginBottom: 12, color: f.color }}>{f.icon}</div>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>{f.title}</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>{f.desc}</div>
            <div style={{ marginTop: 14, fontSize: 12, color: f.color }}>Open →</div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{ gap: 24, marginBottom: 24 }}>
        {/* Severity trend */}
        <div className="card">
          <h2>Severity Trend (Recent Sessions)</h2>
          {stats ? (
            <>
              <TrendChart data={stats.severity_trend} />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 11, color: "var(--text-muted)" }}>
                <span>Older</span><span>Recent</span>
              </div>
            </>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Connect API to see trend</div>
          )}
        </div>

        {/* Disorder distribution */}
        <div className="card">
          <h2>Disorder Distribution</h2>
          {stats?.disorder_distribution?.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {stats.disorder_distribution.slice(0, 6).map(d => {
                const max = stats.disorder_distribution[0].count;
                return (
                  <div key={d.predicted_disorder}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{d.predicted_disorder}</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: "var(--red)", fontFamily: "var(--font-mono)" }}>{d.count}</span>
                    </div>
                    <MiniBar value={d.count} max={max} color="var(--red)" />
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No session data yet</div>
          )}
        </div>
      </div>

      {/* Recent sessions */}
      {stats?.recent_sessions?.length > 0 && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h2>Recent Sessions</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {stats.recent_sessions.map(s => (
              <div key={s.session_id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "var(--bg-deep)", borderRadius: "var(--radius)", border: "1px solid var(--border-soft)" }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{s.name}</span>
                  <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 10 }}>{(s.date || "").split("T")[0]}</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="tag tag-red">{s.predicted_disorder}</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: SEV_COLOR[s.severity_label] || "var(--text-muted)" }}>
                    {s.severity_label}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pipeline + Models */}
      <div className="grid-2" style={{ gap: 24 }}>
        <div className="card">
          <h2>System Pipeline</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {pipeline.map(({ step, label, desc }) => (
              <div key={step} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                <div style={{ color: "var(--accent)", minWidth: 24, fontSize: 11, fontFamily: "var(--font-mono)", paddingTop: 2 }}>{step}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{label}</div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
  <h2>Datasets</h2>
  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
    {/* GoEmotions */}
    <div
      style={{
        padding: 14,
        background: "var(--bg-deep)",
        borderRadius: "var(--radius)",
        border: "1px solid var(--green)30",
      }}
    >
      <div style={{ fontWeight: 600, color: "var(--green)", marginBottom: 4 }}>
        GoEmotions
      </div>
      <div
        style={{
          fontSize: 20,
          fontFamily: "var(--font-display)",
          marginBottom: 4,
        }}
      >
        211,225
      </div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
        28 emotion labels · Emotion Detection
      </div>
    </div>

    {/* Disorder Classification Group */}
    <div
      style={{
        padding: 14,
        background: "var(--bg-deep)",
        borderRadius: "var(--radius)",
        border: "1px solid var(--accent)30",
      }}
    >
      <div style={{ fontWeight: 600, color: "var(--accent)", marginBottom: 4 }}>
        Disorder Classification
      </div>
      <div
        style={{
          fontSize: 20,
          fontFamily: "var(--font-display)",
          marginBottom: 8,
        }}
      >
        5 datasets
      </div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 10 }}>
        Depression, Anxiety, Stress, Insomnia, Panic Disorder
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {[
          { name: "Reddit Depression Dataset", size: "7,731", source: "Kaggle" },
          { name: "CSSRS / Anxiety Reddit", size: "~5,000", source: "HuggingFace" },
          { name: "Dreaddit Stress Dataset", size: "3,553", source: "HuggingFace" },
          { name: "Sleep / Insomnia Reddit Posts", size: "~2,000", source: "HuggingFace" },
          { name: "Panic Disorder Subset", size: "~1,000", source: "Filtered Anxiety Data" },
        ].map((d) => (
          <div
            key={d.name}
            style={{
              padding: "10px 12px",
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 10,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
              {d.name}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
              {d.size} samples · {d.source}
            </div>
          </div>
        ))}
      </div>
    </div>

    {/* NCBI */}
    <div
      style={{
        padding: 14,
        background: "var(--bg-deep)",
        borderRadius: "var(--radius)",
        border: "1px solid var(--purple)30",
      }}
    >
      <div style={{ fontWeight: 600, color: "var(--purple)", marginBottom: 4 }}>
        NCBI Disease
      </div>
      <div
        style={{
          fontSize: 20,
          fontFamily: "var(--font-display)",
          marginBottom: 4,
        }}
      >
        Train + Dev + Test
      </div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
        BIO disease tags · NER Training
      </div>
    </div>
  </div>
</div>
      </div>
    </div>
  );
}