import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

function MetricBar({ value, max = 1, color = "var(--accent)" }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div>
      <div className="progress-bar-bg" style={{ height: 8 }}>
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 3, textAlign: "right" }}>
        {(value * 100).toFixed(1)}%
      </div>
    </div>
  );
}

function ModelCard({ name, data, color }) {
  if (!data) return null;
  const metricFields = [
    { label: "Accuracy",    key: "accuracy",   color: "var(--green)" },
    { label: "Precision",   key: "precision",  color: "var(--accent)" },
    { label: "Recall",      key: "recall",     color: "var(--purple)" },
    { label: "F1 Score",    key: data.f1_macro !== undefined ? "f1_macro" : "f1", color: "var(--yellow)" },
  ];

  return (
    <div className="card" style={{ borderTop: `3px solid ${color}` }}>
      <h2 style={{ color, marginBottom: 4 }}>{name}</h2>
      <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 16 }}>
        {data.model}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {metricFields.map(({ label, key, color: c }) => {
          const val = data[key];
          if (val === undefined) return null;
          return (
            <div key={key}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{label}</span>
                <span style={{ fontSize: 12, fontWeight: 600, color: c }}>
                  {(val * 100).toFixed(1)}%
                </span>
              </div>
              <MetricBar value={val} color={c} />
            </div>
          );
        })}
      </div>

      {data.labels && (
        <div style={{ marginTop: 16 }}>
          <h3>Labels</h3>
          <div className="entity-list">
            {data.labels.map((l) => (
              <span key={l} className="tag tag-blue">{l}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const PLACEHOLDER_METRICS = {
  emotion_model: {
    model: "emotion_model",
    accuracy_subset: 0.82,
    precision_macro: 0.79,
    recall_macro: 0.77,
    f1_macro: 0.78,
    labels: ["fear","sadness","nervousness","anger","joy","neutral","grief","remorse"],
  },
  disorder_model: {
    model: "disorder_model",
    accuracy: 0.91,
    precision: 0.90,
    recall: 0.89,
    f1_macro: 0.895,
    labels: ["Depression","Anxiety","Stress","Insomnia","Panic Disorder"],
  },
  ner_model: {
    model: "ner_model",
    precision: 0.84,
    recall: 0.81,
    f1: 0.825,
  },
  therapy_model: {
    model: "therapy_model",
    accuracy: 0.87,
    precision: 0.86,
    recall: 0.85,
    f1_macro: 0.855,
    labels: [
      "Cognitive Behavioral Therapy","Dialectical Behavior Therapy",
      "Mindfulness-Based Therapy","Exposure Therapy","Sleep Therapy",
      "Psychodynamic Therapy","Group Therapy","Medication Management",
    ],
  },
};

export default function MetricsPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [usingPlaceholder, setUsingPlaceholder] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/metrics`)
      .then((r) => r.json())
      .then((d) => {
        if (d.message) {
          setMetrics(PLACEHOLDER_METRICS);
          setUsingPlaceholder(true);
        } else {
          setMetrics(d);
        }
      })
      .catch(() => {
        setMetrics(PLACEHOLDER_METRICS);
        setUsingPlaceholder(true);
      })
      .finally(() => setLoading(false));
  }, []);

  const modelCards = [
    { key: "emotion_model",  name: "Emotion Detection",        color: "var(--accent)" },
    { key: "disorder_model", name: "Disorder Classifier",      color: "var(--red)" },
    { key: "ner_model",      name: "NER Model (spaCy)",         color: "var(--purple)" },
    { key: "therapy_model",  name: "Therapy Recommender (XGB)", color: "var(--green)" },
  ];

  return (
    <div>
      <h1>Model Evaluation Metrics</h1>
      <p className="page-subtitle">
        Performance metrics for all trained NLP and ML models in the pipeline.
      </p>

      {usingPlaceholder && (
        <div className="alert alert-warn" style={{ marginBottom: 24 }}>
          ⚠ Backend not reachable. Showing expected/target metrics. Train models and start the API to see real results.
        </div>
      )}

      {loading ? (
        <div className="loading-container"><div className="spinner" /></div>
      ) : (
        <>
          <div className="grid-2" style={{ marginBottom: 24 }}>
            {modelCards.map(({ key, name, color }) => (
              <ModelCard
                key={key}
                name={name}
                data={metrics?.[key]}
                color={color}
              />
            ))}
          </div>

          {/* Training info */}
          <div className="card">
            <h2>Training Configuration</h2>
            <div className="grid-3" style={{ gap: 16 }}>
              {[
                { label: "Train / Val / Test Split", value: "70% / 15% / 15%" },
                { label: "Emotion Base Model",       value: "DistilBERT-base-uncased" },
                { label: "Disorder Base Model",      value: "RoBERTa-base" },
                { label: "NER Framework",            value: "spaCy + NCBI Corpus" },
                { label: "Therapy Model",            value: "XGBoost (n=300 trees)" },
                { label: "Evaluation",               value: "Accuracy, P, R, F1, CM" },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  style={{
                    padding: "12px 14px",
                    background: "var(--bg-deep)",
                    borderRadius: "var(--radius)",
                    border: "1px solid var(--border-soft)",
                  }}
                >
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}