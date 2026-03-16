import { useState, useRef } from "react";

const COLORS = {
  patient:    { fill: "#1a3a7a", stroke: "#4f8eff" },
  disorder:   { fill: "#7a1a1a", stroke: "#ff5c5c" },
  symptom:    { fill: "#7a4a0a", stroke: "#f5a623" },
  therapy:    { fill: "#0a5a3a", stroke: "#3ecf8e" },
  medication: { fill: "#3a1a7a", stroke: "#a78bfa" },
  lifestyle:  { fill: "#0a3a5a", stroke: "#38bdf8" },
  default:    { fill: "#2a3a5a", stroke: "#6b7fa3" },
};

const ICONS = {
  patient: "🧠", disorder: "⚠", symptom: "◉",
  therapy: "💬", medication: "💊", lifestyle: "🌿",
};

// ── Tree layout ─────────────────────────────────────────────────
// Column X positions
const COL = { patient: 80, disorder: 240, symptom: 420, therapy: 600, medication: 760, lifestyle: 420 };
const ROW_H = 52;
const PAD_TOP = 40;

function buildTree(nodes, edges) {
  const childMap = {};
  edges.forEach(({ source, target }) => {
    if (!childMap[source]) childMap[source] = [];
    childMap[source].push(target);
  });

  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const pos = {};

  const root = nodes.find(n => n.type === "patient") || nodes[0];
  if (!root) return { pos: {}, height: 200 };

  // Gather by type
  const disorders  = (childMap[root.id] || []).filter(id => nodeMap[id]?.type === "disorder");
  const lifestyles = (childMap[root.id] || []).filter(id => nodeMap[id]?.type === "lifestyle");
  const others1    = (childMap[root.id] || []).filter(id =>
    !disorders.includes(id) && !lifestyles.includes(id));

  let row = 0;

  // Place symptoms + therapy tree under each disorder
  const allSymptoms = [];
  const allTherapies = [];
  const allMeds = [];
  disorders.forEach(did => {
    (childMap[did] || []).forEach(kid => {
      const n = nodeMap[kid];
      if (n?.type === "symptom") allSymptoms.push(kid);
      else if (n?.type === "therapy") allTherapies.push(kid);
    });
  });
  allTherapies.forEach(tid => {
    (childMap[tid] || []).forEach(kid => allMeds.push(kid));
  });

  // Calculate total rows needed
  const maxRows = Math.max(
    disorders.length,
    allSymptoms.length,
    allTherapies.length + allMeds.length,
    lifestyles.length,
    1
  );
  const totalH = PAD_TOP * 2 + maxRows * ROW_H;
  const midY   = totalH / 2;

  // Patient — vertically centered
  pos[root.id] = { x: COL.patient, y: midY };

  // Disorders — centered
  const dStart = midY - ((disorders.length - 1) * ROW_H) / 2;
  disorders.forEach((id, i) => {
    pos[id] = { x: COL.disorder, y: dStart + i * ROW_H };
  });

  // Symptoms — centered
  const sStart = midY - ((allSymptoms.length - 1) * ROW_H) / 2;
  allSymptoms.forEach((id, i) => {
    pos[id] = { x: COL.symptom, y: sStart + i * ROW_H };
  });

  // Therapies + meds
  const tTotal = allTherapies.length + allMeds.length;
  const tStart = midY - ((tTotal - 1) * ROW_H) / 2;
  let tRow = 0;
  allTherapies.forEach(id => {
    pos[id] = { x: COL.therapy, y: tStart + tRow * ROW_H };
    tRow++;
  });
  allMeds.forEach(id => {
    pos[id] = { x: COL.medication, y: tStart + tRow * ROW_H };
    tRow++;
  });

  // Lifestyles — below center-left
  const lStart = midY - ((lifestyles.length - 1) * ROW_H) / 2 + 60;
  lifestyles.forEach((id, i) => {
    pos[id] = { x: COL.lifestyle + 20, y: lStart + i * ROW_H };
  });

  // Others
  others1.forEach((id, i) => {
    pos[id] = pos[id] || { x: COL.disorder, y: midY + 100 + i * ROW_H };
  });

  // Fallback
  nodes.forEach((n, i) => {
    if (!pos[n.id]) pos[n.id] = { x: 100 + (i % 4) * 160, y: 100 + Math.floor(i / 4) * ROW_H };
  });

  return { pos, height: Math.max(totalH, 200) };
}

// Smooth step connector (like flowchart)
function stepPath(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

// Node pill width based on label length
function pillW(label) {
  return Math.min(Math.max(label.length * 7 + 24, 80), 160);
}

export default function MindMap({ data }) {
  const [hovered, setHovered] = useState(null);
  const svgRef = useRef(null);

  const downloadPNG = () => {
    const svg = svgRef.current;
    if (!svg) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svg);
    const blob = new Blob([svgStr], { type: "image/svg+xml" });
    const url  = URL.createObjectURL(blob);
    // Convert SVG to PNG via canvas
    const img  = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const scale  = 2; // retina
      canvas.width  = svg.viewBox.baseVal.width  * scale;
      canvas.height = svg.viewBox.baseVal.height * scale;
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.fillStyle = "#07090e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      const pngUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = "mindmap.png";
      a.click();
    };
    img.src = url;
  };

  if (!data?.nodes?.length) {
    return (
      <div style={{
        padding: 48, textAlign: "center", color: "#4a5572",
        fontSize: 13, background: "#08090e",
        borderRadius: 10, border: "1px dashed #1e2d48",
      }}>
        Analyze a consultation to generate the mind map
      </div>
    );
  }

  const { nodes, edges } = data;
  const { pos, height }  = buildTree(nodes, edges);
  const W = 880;
  const H = Math.max(height, 260);

  const connected = hovered
    ? new Set(edges.flatMap(e =>
        e.source === hovered ? [e.target] :
        e.target === hovered ? [e.source] : []
      ))
    : new Set();

  return (
    <div style={{
      background: "#07090e", borderRadius: 12,
      border: "1px solid #1a2540", overflow: "hidden",
    }}>
      {/* Top bar */}
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between",
        padding: "9px 16px",
        borderBottom: "1px solid #121c2e",
        background: "rgba(255,255,255,0.015)",
      }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {Object.entries(COLORS).filter(([k]) => k !== "default").map(([type, cfg]) => {
            if (!nodes.some(n => n.type === type)) return null;
            return (
              <div key={type} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: cfg.stroke }} />
                <span style={{ color: "#5a6a8a", textTransform: "capitalize" }}>{type}</span>
              </div>
            );
          })}
        </div>
        <button
          onClick={downloadPNG}
          style={{
            background: "var(--accent-soft, rgba(79,142,255,0.12))",
            border: "1px solid #4f8eff55", color: "#4f8eff",
            borderRadius: 6, padding: "5px 12px", cursor: "pointer",
            fontSize: 11, fontFamily: "inherit", fontWeight: 500,
            display: "flex", alignItems: "center", gap: 5,
          }}
        >
          ⬇ PNG
        </button>
      </div>

      {/* Column headers */}
      <div style={{
        display: "flex", padding: "6px 0 0",
        borderBottom: "1px solid #0d1525",
      }}>
        {[
          { label: "Patient",  x: COL.patient },
          { label: "Disorder", x: COL.disorder },
          { label: "Symptoms", x: COL.symptom },
          { label: "Therapy",  x: COL.therapy },
        ].map(({ label, x }) => (
          nodes.some(n => n.type === label.toLowerCase() ||
            (label === "Symptoms" && n.type === "symptom") ||
            (label === "Therapy" && (n.type === "therapy" || n.type === "medication"))
          ) ? (
            <div key={label} style={{
              position: "absolute",
              left: x, fontSize: 9, color: "#2a3a55",
              fontWeight: 700, letterSpacing: 1,
              textTransform: "uppercase",
            }}>
              {label}
            </div>
          ) : null
        ))}
      </div>

      {/* SVG */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        style={{ display: "block", overflow: "visible" }}
      >
        <defs>
          {Object.entries(COLORS).map(([type, cfg]) => (
            <linearGradient key={type} id={`lg_${type}`} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor={cfg.stroke} stopOpacity="0.9" />
              <stop offset="100%" stopColor={cfg.fill}   stopOpacity="0.9" />
            </linearGradient>
          ))}
          <filter id="mm_glow">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="mm_sm">
            <feGaussianBlur stdDeviation="1.5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <pattern id="mm_dots" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="0.7" fill="#fff" opacity="0.022" />
          </pattern>
        </defs>

        <rect width={W} height={H} fill="#07090e" />
        <rect width={W} height={H} fill="url(#mm_dots)" />

        {/* Column separator lines */}
        {[COL.disorder - 15, COL.symptom - 15, COL.therapy - 15].map(x => (
          <line key={x} x1={x} y1={20} x2={x} y2={H - 20}
            stroke="#0f1a2e" strokeWidth="1" strokeDasharray="4 4" />
        ))}

        {/* Edges */}
        {edges.map((e, i) => {
          const sp = pos[e.source];
          const tp = pos[e.target];
          if (!sp || !tp) return null;

          const sn  = nodes.find(n => n.id === e.source);
          const col = (COLORS[sn?.type] || COLORS.default).stroke;
          const active = hovered === e.source || hovered === e.target;
          const dimmed = hovered && !active;
          const sw   = pillW(nodes.find(n => n.id === e.source)?.label || "");

          return (
            <path key={i}
              d={stepPath(sp.x + sw / 2, sp.y, tp.x - pillW(nodes.find(n => n.id === e.target)?.label || "") / 2, tp.y)}
              fill="none"
              stroke={active ? col : "#1e2d48"}
              strokeWidth={active ? 2 : 1}
              opacity={dimmed ? 0.08 : active ? 1 : 0.45}
              strokeLinecap="round"
            />
          );
        })}

        {/* Nodes */}
        {nodes.map(node => {
          const p   = pos[node.id];
          if (!p) return null;

          const cfg    = COLORS[node.type] || COLORS.default;
          const isRoot = node.type === "patient";
          const isHov  = hovered === node.id;
          const isCon  = connected.has(node.id);
          const dimmed = hovered && !isHov && !isCon;
          const label  = node.label || node.id;
          const w      = pillW(label);
          const h      = isRoot ? 38 : 30;
          const rx     = h / 2;
          const icon   = ICONS[node.type] || "";

          return (
            <g
              key={node.id}
              transform={`translate(${p.x - w / 2}, ${p.y - h / 2})`}
              style={{ cursor: "pointer", opacity: dimmed ? 0.15 : 1 }}
              onMouseEnter={() => setHovered(node.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Glow */}
              {(isHov || isRoot) && (
                <rect x={-4} y={-4} width={w + 8} height={h + 8}
                  rx={rx + 4}
                  fill={cfg.stroke} opacity={0.12}
                />
              )}
              {/* Shadow */}
              <rect x={2} y={3} width={w} height={h} rx={rx}
                fill="#000" opacity={0.4} />
              {/* Main pill */}
              <rect width={w} height={h} rx={rx}
                fill={`url(#lg_${node.type || "default"})`}
                stroke={cfg.stroke}
                strokeWidth={isRoot ? 2 : isHov ? 1.8 : 1.2}
                strokeOpacity={isHov ? 1 : 0.6}
                filter={isRoot || isHov ? "url(#mm_glow)" : "url(#mm_sm)"}
              />
              {/* Shine */}
              <rect x={6} y={3} width={w - 12} height={h / 3}
                rx={h / 6} fill="#fff" opacity={0.1} />
              {/* Icon */}
              {icon && (
                <text x={10} y={h / 2 + 4}
                  fontSize={isRoot ? 13 : 11}
                  style={{ pointerEvents: "none", userSelect: "none" }}>
                  {icon}
                </text>
              )}
              {/* Label */}
              <text
                x={icon ? 26 : w / 2}
                y={h / 2 + 1}
                textAnchor={icon ? "start" : "middle"}
                dominantBaseline="middle"
                fill="#fff"
                fontSize={isRoot ? 11 : label.length > 14 ? 8.5 : 9.5}
                fontWeight={isRoot ? "700" : "600"}
                fontFamily="'Outfit', 'Segoe UI', sans-serif"
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                {label.length > 20 ? label.slice(0, 18) + "…" : label}
              </text>
            </g>
          );
        })}

        {/* Hover tooltip */}
        {hovered && (() => {
          const n = nodes.find(n => n.id === hovered);
          const p = pos[hovered];
          if (!n || !p) return null;
          const cfg = COLORS[n.type] || COLORS.default;
          const ty  = Math.max(p.y - 36, 8);
          const tx  = Math.min(p.x + pillW(n.label) / 2 + 8, W - 170);
          return (
            <g>
              <rect x={tx} y={ty} width={160} height={30}
                rx="6" fill="#0d1525"
                stroke={cfg.stroke} strokeWidth="1" strokeOpacity="0.6"
              />
              <text x={tx + 10} y={ty + 11}
                fill={cfg.stroke} fontSize="8" fontWeight="700"
                fontFamily="'Outfit', sans-serif" letterSpacing="0.8">
                {(n.type || "").toUpperCase()}
              </text>
              <text x={tx + 10} y={ty + 23}
                fill="#e8edf5" fontSize="10" fontWeight="600"
                fontFamily="'Outfit', sans-serif">
                {n.label}
              </text>
            </g>
          );
        })()}
      </svg>

      <div style={{
        padding: "7px 16px", borderTop: "1px solid #111c2e",
        fontSize: 10, color: "#2a3a55", textAlign: "right",
        background: "rgba(0,0,0,0.25)",
      }}>
        Hover to highlight · ⬇ PNG to download
      </div>
    </div>
  );
}