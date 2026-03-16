import { useState, useRef } from "react";
import MindMap from "../components/MindMap";

const API_BASE = "http://localhost:8000";

const SAMPLE_TEXT = `Patient presents with significant anxiety and restlessness, 
particularly before academic examinations. Reports difficulty sleeping, 
persistent overthinking, and loss of concentration. Patient has experienced 
mild panic attacks with chest tightness. Social withdrawal observed. 
CBT has been suggested. Currently prescribed sertraline 50mg. 
Mindfulness exercises and regular physical activity recommended.`;

const SEVERITY_CONFIG = {
  Low:    { color: "var(--green)",  bg: "var(--green-soft)",  bar: "#3ecf8e" },
  Medium: { color: "var(--yellow)", bg: "var(--yellow-soft)", bar: "#f5a623" },
  High:   { color: "var(--red)",    bg: "var(--red-soft)",    bar: "#ff5c5c" },
};

const SEV_STYLE = {
  Low:    { color: "#3ecf8e", bg: "rgba(62,207,142,0.08)",  bar: "#3ecf8e" },
  Medium: { color: "#f5a623", bg: "rgba(245,166,35,0.08)",  bar: "#f5a623" },
  High:   { color: "#ff5c5c", bg: "rgba(255,92,92,0.08)",   bar: "#ff5c5c" },
};

const DISORDER_ICONS = {
  "Depression": "😔", "Anxiety": "😰", "Stress": "😤",
  "Insomnia": "🌙", "Panic Disorder": "💨",
  "Performance Anxiety": "📚", "PTSD": "🧩", "default": "🧠",
};

const THERAPY_ICONS = {
  "Cognitive Behavioral Therapy": "💬",
  "Dialectical Behavior Therapy": "⚖️",
  "Mindfulness-Based Therapy": "🧘",
  "Exposure Therapy": "🌅",
  "Sleep Therapy": "😴",
  "Psychodynamic Therapy": "🔍",
  "Group Therapy": "👥",
  "EMDR Therapy": "👁️",
  "Medication Management": "💊",
  "default": "✨",
};

const THERAPY_DESCS = {
  "Cognitive Behavioral Therapy":   "Identifies and changes negative thought patterns through structured, evidence-based techniques.",
  "Dialectical Behavior Therapy":   "Builds skills in emotional regulation, mindfulness, and healthy relationships.",
  "Mindfulness-Based Therapy":      "Cultivates present-moment awareness to reduce anxiety and emotional reactivity.",
  "Exposure Therapy":               "Gradually faces feared situations to reduce avoidance and build confidence.",
  "Sleep Therapy":                  "Restores healthy sleep patterns using behavioural and cognitive strategies.",
  "Psychodynamic Therapy":          "Explores past experiences to resolve present emotional difficulties.",
  "Group Therapy":                  "Supportive group environment where shared experiences foster healing.",
  "EMDR Therapy":                   "Processes traumatic memories through eye movement techniques.",
  "Medication Management":          "Carefully monitored use of prescribed medication alongside therapy.",
};

// ── Report sub-components ─────────────────────────────────────────

function SeverityBar({ score, label }) {
  const cfg = SEV_STYLE[label] || SEV_STYLE.Low;
  const pct = Math.min((score / 3) * 100, 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Low</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: cfg.color }}>
          {label} — {score?.toFixed(1)} / 3.0
        </span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>High</span>
      </div>
      <div style={{ height: 10, background: "var(--bg-deep)", borderRadius: 6, overflow: "hidden", border: "1px solid var(--border-soft)" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg, ${cfg.bar}88, ${cfg.bar})`, borderRadius: 6, boxShadow: `0 0 8px ${cfg.bar}60`, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

function Section({ title, icon, color = "var(--accent)", children }) {
  return (
    <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden", marginBottom: 16 }}>
      <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border-soft)", background: `linear-gradient(90deg, ${color}15, transparent)`, display: "flex", alignItems: "center", gap: 10, borderLeft: `3px solid ${color}` }}>
        {icon && <span style={{ fontSize: 18 }}>{icon}</span>}
        <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1.2, color }}>{title}</span>
      </div>
      <div style={{ padding: "16px 20px" }}>{children}</div>
    </div>
  );
}

function BulletList({ items, color = "var(--accent)" }) {
  if (!items?.length) return <span style={{ color: "var(--text-muted)", fontSize: 13 }}>None identified</span>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {items.map((item, i) => (
        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, marginTop: 6, flexShrink: 0, boxShadow: `0 0 5px ${color}80` }} />
          <span style={{ fontSize: 13.5, color: "var(--text-primary)", lineHeight: 1.6 }}>{item}</span>
        </div>
      ))}
    </div>
  );
}

function InfoRow({ label, value }) {
  if (!value) return null;
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid var(--border-soft)" }}>
      <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 500 }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

function ReportView({ result }) {
  if (!result) return null;
  const sev       = SEV_STYLE[result.severity_label] || SEV_STYLE.Low;
  const firstName = (result.patient_name || "Patient").split(" ")[0];
  const entities  = result.entities || {};
  const dIcon     = DISORDER_ICONS[result.predicted_disorder] || DISORDER_ICONS.default;
  const tIcon     = THERAPY_ICONS[result.recommended_therapy]  || THERAPY_ICONS.default;
  const tDesc     = THERAPY_DESCS[result.recommended_therapy]  || "A personalised therapeutic approach.";

  const sevMsg = {
    Low:    `${firstName}, your emotional state is relatively stable. With the right support, things will continue to improve.`,
    Medium: `${firstName}, you are experiencing moderate distress. Consistent therapy and self-care will make a significant difference.`,
    High:   `${firstName}, you are experiencing significant distress. Professional support is strongly recommended.`,
  }[result.severity_label] || "";

  const handlePrint = () => {
    const el = document.getElementById("psych-report-content");
    if (!el) return;
    const win = window.open("", "_blank");
    win.document.write(`<!DOCTYPE html><html><head><title>Report - ${result.patient_name}</title>
      <style>body{font-family:'Segoe UI',Arial,sans-serif;background:#fff;color:#1a1a2e;margin:0;padding:20px;}
      *{box-sizing:border-box;}.no-print{display:none!important;}</style></head>
      <body>${el.innerHTML}<script>window.onload=()=>{window.print();window.close();}<\/script></body></html>`);
    win.document.close();
  };

  const handleTxt = () => {
    const blob = new Blob([result.report || ""], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `report_${result.patient_name || "patient"}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #0d1525 0%, #111e35 100%)", border: "1px solid var(--border)", borderRadius: 14, padding: "28px 32px", marginBottom: 20, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -30, right: -30, width: 150, height: 150, borderRadius: "50%", background: "var(--accent)", opacity: 0.04 }} />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
          <div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: 2, marginBottom: 6, textTransform: "uppercase" }}>Psychological Consultation Report</div>
            <div style={{ fontFamily: "var(--font-display)", fontSize: 26, color: "var(--text-primary)", marginBottom: 8 }}>Dear {firstName},</div>
            <div style={{ fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: 480 }}>
              Thank you for your consultation. This report summarises your session findings and outlines a personalised plan to support your wellbeing.
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={handlePrint} style={{ background: "var(--accent-soft)", border: "1px solid var(--accent)", color: "var(--accent)", borderRadius: 8, padding: "8px 14px", cursor: "pointer", fontSize: 12, fontFamily: "inherit", fontWeight: 500 }}>
              🖨 Print / PDF
            </button>
            <button onClick={handleTxt} style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-secondary)", borderRadius: 8, padding: "8px 14px", cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>
              ⬇ TXT
            </button>
          </div>
        </div>
      </div>

      {/* Printable content */}
      <div id="psych-report-content">

        {/* Patient + Severity */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          <Section title="Patient Details" icon="👤" color="var(--accent)">
            <InfoRow label="Full Name"         value={result.patient_name} />
            <InfoRow label="Age"               value={result.patient_age} />
            <InfoRow label="Gender"            value={result.patient_gender} />
            <InfoRow label="Consultation Date" value={result.consultation_date} />
            <InfoRow label="Session ID"        value={result.session_id} />
          </Section>
          <Section title="Severity Assessment" icon="📊" color={sev.color}>
            <div style={{ marginBottom: 16 }}>
              <SeverityBar score={result.severity_score} label={result.severity_label} />
            </div>
            {result.clinical_scale && (
              <div style={{ padding: "8px 12px", background: `${sev.color}15`, border: `1px solid ${sev.color}30`, borderRadius: 8, fontSize: 12, color: sev.color, fontWeight: 600, marginBottom: 10 }}>
                {result.clinical_scale.scale}: ~{result.clinical_scale.estimated_score} / {result.clinical_scale.range.split("–")[1]} — {result.clinical_scale.band}
              </div>
            )}
            <div style={{ background: sev.bg, border: `1px solid ${sev.color}30`, borderRadius: 8, padding: "12px 14px", fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
              {sevMsg}
            </div>
          </Section>
        </div>

        {/* Disorder */}
        <Section title="Identified Concern" icon={dIcon} color="#ff5c5c">
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ padding: "12px 24px", background: "rgba(255,92,92,0.1)", border: "1px solid rgba(255,92,92,0.3)", borderRadius: 10, fontSize: 18, fontWeight: 700, color: "#ff5c5c", whiteSpace: "nowrap" }}>
              {result.predicted_disorder}
            </div>
            {result.disorder_confidence && (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 80, height: 6, background: "var(--bg-deep)", borderRadius: 4, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${result.disorder_confidence}%`, background: result.disorder_confidence >= 80 ? "var(--green)" : result.disorder_confidence >= 60 ? "var(--yellow)" : "var(--red)", borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 12, fontWeight: 700, color: result.disorder_confidence >= 80 ? "var(--green)" : result.disorder_confidence >= 60 ? "var(--yellow)" : "var(--red)" }}>
                  {result.disorder_confidence}% confidence
                </span>
              </div>
            )}
          </div>
        </Section>

        {/* Emotions */}
        <Section title="Emotional State" icon="💭" color="#a78bfa">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {(result.emotions || []).length > 0
              ? (result.emotions || []).map(e => (
                  <span key={e} style={{ padding: "5px 14px", borderRadius: 20, background: "rgba(167,139,250,0.12)", border: "1px solid rgba(167,139,250,0.3)", color: "#a78bfa", fontSize: 13, fontWeight: 500, textTransform: "capitalize" }}>
                    {e}
                  </span>
                ))
              : <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Neutral / Stable</span>
            }
          </div>
        </Section>

        {/* Symptoms + Lifestyle */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          <Section title="Symptoms Observed" icon="◉" color="#f5a623">
            <BulletList items={entities.SYMPTOM} color="#f5a623" />
          </Section>
          <Section title="Lifestyle Recommendations" icon="🌿" color="#38bdf8">
            <BulletList items={entities.LIFESTYLE} color="#38bdf8" />
          </Section>
        </div>

        {/* Therapy */}
        <Section title="Recommended Therapy" icon={tIcon} color="#3ecf8e">
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ padding: "14px 22px", background: "rgba(62,207,142,0.1)", border: "1px solid rgba(62,207,142,0.3)", borderRadius: 10, fontSize: 16, fontWeight: 700, color: "#3ecf8e", whiteSpace: "nowrap" }}>
              {tIcon} {result.recommended_therapy}
            </div>
            <div style={{ fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.8, flex: 1 }}>{tDesc}</div>
          </div>
        </Section>

        {/* Medications */}
        {entities.MEDICATION?.length > 0 && (
          <Section title="Medications Prescribed" icon="💊" color="#a78bfa">
            <BulletList items={entities.MEDICATION} color="#a78bfa" />
          </Section>
        )}

        {/* Remarks */}
        {result.remarks && (
          <Section title="Clinician Remarks" icon="📝" color="var(--accent)">
            <div style={{ background: "var(--accent-soft)", border: "1px solid rgba(79,142,255,0.2)", borderRadius: 8, padding: "14px 18px", fontSize: 14, color: "var(--text-primary)", lineHeight: 1.8, fontStyle: "italic" }}>
              "{result.remarks}"
            </div>
          </Section>
        )}

        {/* Closing */}
        <div style={{ background: "linear-gradient(135deg, #0a1525 0%, #0d1e30 100%)", border: "1px solid var(--border)", borderRadius: 12, padding: "24px 28px", textAlign: "center" }}>
          <div style={{ fontSize: 22, marginBottom: 10 }}>💙</div>
          <div style={{ fontFamily: "var(--font-display)", fontSize: 18, color: "var(--text-primary)", marginBottom: 10 }}>You are not alone in this journey.</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8, maxWidth: 500, margin: "0 auto" }}>
            {firstName}, seeking support is a sign of strength, not weakness. Every step you take matters.
            <br />
            <span style={{ color: "var(--accent)", fontWeight: 600 }}>Progress, not perfection. Be kind to yourself.</span>
          </div>
          <div style={{ marginTop: 16, fontSize: 11, color: "var(--text-muted)", borderTop: "1px solid var(--border-soft)", paddingTop: 12 }}>
            Generated {new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}
            &nbsp;·&nbsp; NLP-based analysis &nbsp;·&nbsp;
            Consult a licensed professional for clinical decisions
          </div>
        </div>

      </div>
    </div>
  );
}

// ── Main AnalysisPage ────────────────────────────────────────────

export default function AnalysisPage({ result, setResult }) {
  const [inputMode,   setInputMode]   = useState("text");
  const [text,        setText]        = useState("");
  const [pdfFile,     setPdfFile]     = useState(null);
  const [dragOver,    setDragOver]    = useState(false);
  const [patientName, setPatientName] = useState("");
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [saved,       setSaved]       = useState(false);
  const [activeTab,   setActiveTab]   = useState("overview");
  const fileInputRef = useRef(null);

  const handleAnalyzeText = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(null); setSaved(false);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, patient_name: patientName || "Anonymous Patient" }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API error ${res.status}`);
      }
      setResult(await res.json());
      setActiveTab("overview");
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const handleAnalyzePdf = async () => {
    if (!pdfFile) return;
    setLoading(true); setError(null); setSaved(false);
    try {
      const fd = new FormData();
      fd.append("file", pdfFile);
      fd.append("patient_name", patientName || "Anonymous Patient");
      const res = await fetch(`${API_BASE}/analyze-pdf`, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API error ${res.status}`);
      }
      setResult(await res.json());
      setActiveTab("overview");
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const handleSave = async () => {
    if (!result) return;
    const pid = (patientName || "anonymous").replace(/\s+/g, "_").toLowerCase();
    try {
      const res = await fetch(`${API_BASE}/add-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: pid, session_data: result }),
      });
      if (!res.ok) throw new Error("Save failed");
      setSaved(pid);
    } catch { setError("Failed to save session"); }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/pdf") { setPdfFile(file); setError(null); }
    else setError("Only PDF files are supported.");
  };

  const canAnalyze = inputMode === "text" ? text.trim().length > 0 : pdfFile !== null;
  const severity   = result?.severity_label ? SEVERITY_CONFIG[result.severity_label] || SEVERITY_CONFIG.Low : null;
  const tabs       = ["overview", "entities", "mindmap", "report"];

  return (
    <div>
      <h1>Consultation Analysis</h1>
      <p className="page-subtitle">Analyse psychological consultation notes via text input or PDF upload.</p>

      {/* Input card */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="grid-2" style={{ gap: 20, marginBottom: 20 }}>
          <div>
            <label>Patient Name (optional)</label>
            <input type="text" placeholder="e.g. Jane Doe" value={patientName} onChange={e => setPatientName(e.target.value)} />
          </div>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
            {["text","pdf"].map(mode => (
              <button key={mode} className={`btn ${inputMode === mode ? "btn-primary" : "btn-ghost"}`}
                style={{ flex: 1 }} onClick={() => { setInputMode(mode); setError(null); }}>
                {mode === "text" ? "✏ Text Input" : "📄 PDF Upload"}
              </button>
            ))}
          </div>
        </div>

        {inputMode === "text" && (
          <>
            <label>Consultation Text</label>
            <textarea rows={8} placeholder="Paste psychological consultation notes here…" value={text} onChange={e => setText(e.target.value)} />
            <div style={{ marginTop: 8, textAlign: "right" }}>
              <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => setText(SAMPLE_TEXT)}>Load Sample</button>
            </div>
          </>
        )}

        {inputMode === "pdf" && (
          <div>
            <label>Upload PDF</label>
            <div onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
              style={{ border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`, borderRadius: "var(--radius)", background: dragOver ? "var(--accent-soft)" : "var(--bg-deep)", padding: "40px 20px", textAlign: "center", cursor: "pointer", transition: "all 0.2s" }}>
              {pdfFile
                ? <div><div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
                    <div style={{ fontWeight: 600, color: "var(--green)", marginBottom: 4 }}>{pdfFile.name}</div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{(pdfFile.size/1024).toFixed(1)} KB</div>
                  </div>
                : <div><div style={{ fontSize: 36, marginBottom: 10 }}>📂</div>
                    <div style={{ color: "var(--text-secondary)", marginBottom: 4 }}>Drag & drop a PDF, or click to browse</div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Consultation notes, prescriptions, reports (.pdf)</div>
                  </div>
              }
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" style={{ display: "none" }} onChange={e => { if (e.target.files[0]) { setPdfFile(e.target.files[0]); setError(null); }}} />
            {pdfFile && (
              <div style={{ marginTop: 8, textAlign: "right" }}>
                <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={e => { e.stopPropagation(); setPdfFile(null); }}>✕ Remove</button>
              </div>
            )}
          </div>
        )}

        <div className="flex-row" style={{ marginTop: 16 }}>
          <button className="btn btn-primary" onClick={inputMode === "text" ? handleAnalyzeText : handleAnalyzePdf} disabled={loading || !canAnalyze}>
            {loading ? "Analyzing…" : inputMode === "text" ? "◈  Analyze Text" : "◈  Analyze PDF"}
          </button>
          {result && !loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <button className="btn btn-ghost" onClick={handleSave} disabled={!!saved}>
                {saved ? "✓ Saved" : "💾 Save Session"}
              </button>
              {saved && <div style={{ fontSize: 11, color: "var(--green)", fontFamily: "var(--font-mono)", textAlign: "center" }}>ID: {saved}</div>}
            </div>
          )}
        </div>
        {error && <div className="alert alert-error" style={{ marginTop: 12 }}>✕ {error}</div>}
      </div>

      {loading && <div className="loading-container"><div className="spinner" /><span>Running NLP pipeline…</span></div>}

      {result && !loading && (
        <div className="fade-up">

          {/* Validation warning */}
          {result.input_validation && !result.input_validation.is_valid && (
            <div className="alert alert-error" style={{ marginBottom: 16 }}>
              ⚠ {result.input_validation.warning || result.input_validation.reason}
              <span style={{ display: "block", fontSize: 11, marginTop: 4, opacity: 0.8 }}>Results shown may be inaccurate. Provide proper consultation notes for best results.</span>
            </div>
          )}
          {result.input_validation?.is_valid && result.input_validation?.warning && (
            <div className="alert alert-warn" style={{ marginBottom: 16 }}>ℹ {result.input_validation.warning}</div>
          )}
          {result.mode === "rule-based" && (
            <div className="alert alert-warn" style={{ marginBottom: 16 }}>⚠ Rule-based fallback mode. Train ML models for better predictions.</div>
          )}

          {/* Feature 10: Risk flag */}
          {result.risk?.flag && (
            <div style={{
              padding: "14px 18px", borderRadius: 10, marginBottom: 16,
              background: result.risk.level === "HIGH" ? "rgba(255,50,50,0.12)" : "rgba(255,140,0,0.1)",
              border: `2px solid ${result.risk.level === "HIGH" ? "#ff3232" : "#ff8c00"}`,
              borderLeft: `5px solid ${result.risk.level === "HIGH" ? "#ff3232" : "#ff8c00"}`,
            }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: result.risk.level === "HIGH" ? "#ff3232" : "#ff8c00", marginBottom: 6 }}>
                {result.risk.level === "HIGH" ? "🚨 URGENT: High-Risk Indicators Detected" : "⚠ Risk Indicators Detected"}
              </div>
              <div style={{ fontSize: 13, color: "var(--text-primary)", marginBottom: 6 }}>
                {result.risk.message}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8 }}>
                {result.risk.action}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {result.risk.keywords?.map(kw => (
                  <span key={kw} style={{ padding: "2px 8px", borderRadius: 10, background: "rgba(255,50,50,0.15)", border: "1px solid rgba(255,50,50,0.3)", color: "#ff6666", fontSize: 11 }}>
                    "{kw}"
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Feature 9: Therapy explanation */}
          {result.therapy_explanation && (
            <div style={{ padding: "12px 16px", borderRadius: 8, marginBottom: 16, background: "rgba(62,207,142,0.06)", border: "1px solid rgba(62,207,142,0.2)" }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#3ecf8e", marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.8 }}>
                💡 Why this therapy?
              </div>
              <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
                {result.therapy_explanation}
              </div>
            </div>
          )}

          {/* Stat cards */}
          <div className="grid-4" style={{ marginBottom: 24 }}>

            {/* Disorder + confidence */}
            <div className="stat-card">
              <div className="stat-label">Predicted Disorder</div>
              <div className="stat-value" style={{ fontSize: 15, color: "var(--red)", marginBottom: 6 }}>{result.predicted_disorder}</div>
              {result.disorder_confidence && (
                <>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ flex: 1, height: 4, background: "var(--bg-deep)", borderRadius: 4, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${result.disorder_confidence}%`, background: result.disorder_confidence >= 80 ? "var(--green)" : result.disorder_confidence >= 60 ? "var(--yellow)" : "var(--red)", borderRadius: 4 }} />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 700, fontFamily: "var(--font-mono)", color: result.disorder_confidence >= 80 ? "var(--green)" : result.disorder_confidence >= 60 ? "var(--yellow)" : "var(--red)" }}>
                      {result.disorder_confidence}%
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>
                    {result.disorder_confidence >= 80 ? "High confidence" : result.disorder_confidence >= 60 ? "Moderate — verify clinically" : "Low — manual review recommended"}
                  </div>
                </>
              )}
            </div>

            {/* Severity */}
            <div className="stat-card" style={{ background: severity?.bg, borderColor: (severity?.color || "var(--border)") + "40" }}>
              <div className="stat-label">Severity</div>
              <div className="stat-value" style={{ color: severity?.color, fontSize: 22 }}>{result.severity_label}</div>
              <div className="stat-sub">Score: {result.severity_score?.toFixed(2)} / 3.0</div>
              <div className="progress-bar-bg" style={{ marginTop: 8 }}>
                <div className="progress-bar-fill" style={{ width: `${((result.severity_score || 0) / 3) * 100}%`, background: severity?.bar }} />
              </div>
              {result.clinical_scale && (
                <div style={{ marginTop: 8, padding: "5px 8px", background: "rgba(0,0,0,0.2)", borderRadius: 6, fontSize: 10, color: severity?.color, fontFamily: "var(--font-mono)" }}>
                  {result.clinical_scale.scale}: ~{result.clinical_scale.estimated_score}/{result.clinical_scale.range.split("–")[1]}
                </div>
              )}
            </div>

            {/* Therapy */}
            <div className="stat-card">
              <div className="stat-label">Recommended Therapy</div>
              <div className="stat-value" style={{ fontSize: 13, color: "var(--green)", paddingTop: 4 }}>{result.recommended_therapy}</div>
            </div>

            {/* Emotions */}
            <div className="stat-card">
              <div className="stat-label">Emotions Detected</div>
              <div className="stat-value" style={{ fontSize: 22 }}>{result.emotions?.length || 0}</div>
              <div className="stat-sub">{result.emotions?.slice(0,3).join(", ")}</div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid var(--border)" }}>
            {tabs.map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={{ background: "none", border: "none", padding: "8px 16px", color: activeTab === t ? "var(--accent)" : "var(--text-secondary)", borderBottom: activeTab === t ? "2px solid var(--accent)" : "2px solid transparent", cursor: "pointer", fontSize: 13.5, fontFamily: "var(--font-body)", fontWeight: activeTab === t ? 600 : 400, textTransform: "capitalize", transition: "color 0.15s" }}>
                {t}
              </button>
            ))}
          </div>

          {/* Overview */}
          {activeTab === "overview" && (
            <div className="grid-2" style={{ gap: 20 }}>
              <div className="card">
                <h2>Detected Emotions</h2>
                <div className="entity-list">
                  {result.emotions?.length > 0
                    ? result.emotions.map(e => <span key={e} className="tag tag-blue">{e}</span>)
                    : <span style={{ color: "var(--text-muted)", fontSize: 13 }}>None detected</span>}
                </div>
              </div>
              <div className="card">
                <h2>Session Summary</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    { k: "Session ID",   v: result.session_id },
                    { k: "Patient",      v: result.patient_name },
                    result.patient_age       ? { k: "Age",          v: result.patient_age }       : null,
                    result.patient_gender    ? { k: "Gender",       v: result.patient_gender }     : null,
                    result.consultation_date ? { k: "Consult Date", v: result.consultation_date }  : null,
                    { k: "Disorder",     v: result.predicted_disorder },
                    { k: "Therapy",      v: result.recommended_therapy },
                    { k: "Severity",     v: `${result.severity_label} (${result.severity_score?.toFixed(2)})` },
                    { k: "Mode",         v: result.mode },
                    result.remarks ? { k: "Remarks", v: result.remarks } : null,
                  ].filter(Boolean).map(({ k, v }) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", paddingBottom: 8, borderBottom: "1px solid var(--border-soft)" }}>
                      <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>{k}</span>
                      <span style={{ fontSize: 13, fontWeight: 500 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Entities */}
          {activeTab === "entities" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {Object.entries(result.entities || {}).map(([type, items]) => (
                <div key={type} className="card card-sm">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <h3 style={{ margin: 0 }}>{type}</h3>
                    <span className="tag tag-blue">{items.length}</span>
                  </div>
                  <div className="entity-list">
                    {items.length > 0
                      ? items.map(item => (
                          <span key={item} className={`tag ${type==="SYMPTOM"?"tag-yellow":type==="DISORDER"?"tag-red":type==="THERAPY"?"tag-green":type==="MEDICATION"?"tag-purple":"tag-blue"}`}>{item}</span>
                        ))
                      : <span style={{ color: "var(--text-muted)", fontSize: 12 }}>None detected</span>
                    }
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Mind Map */}
          {activeTab === "mindmap" && (
            <div className="card">
              <h2>Dynamic Mind Map</h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 16 }}>
                Visual relationships between patient, symptoms, disorder, therapy, medications and lifestyle.
              </p>
              <MindMap data={result.mind_map} />
            </div>
          )}

          {/* Report */}
          {activeTab === "report" && (
            <ReportView result={result} />
          )}

        </div>
      )}
    </div>
  );
}