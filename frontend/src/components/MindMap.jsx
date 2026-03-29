import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ── Design tokens ────────────────────────────────────────────────────────────
const COLORS = {
  patient:    { bg: "#0a1628", border: "#3b82f6", accent: "#60a5fa", glow: "rgba(59,130,246,0.35)", badge: "#1d3a6e", text: "#bfdbfe" },
  disorder:   { bg: "#1a0a0a", border: "#ef4444", accent: "#f87171", glow: "rgba(239,68,68,0.35)",  badge: "#6b1f1f", text: "#fecaca" },
  symptom:    { bg: "#1a1200", border: "#f59e0b", accent: "#fbbf24", glow: "rgba(245,158,11,0.35)", badge: "#6b4e00", text: "#fef3c7" },
  therapy:    { bg: "#021a10", border: "#10b981", accent: "#34d399", glow: "rgba(16,185,129,0.35)", badge: "#064e36", text: "#a7f3d0" },
  medication: { bg: "#100a1a", border: "#8b5cf6", accent: "#a78bfa", glow: "rgba(139,92,246,0.35)", badge: "#3b1f6e", text: "#ede9fe" },
  lifestyle:  { bg: "#001a1a", border: "#06b6d4", accent: "#22d3ee", glow: "rgba(6,182,212,0.35)",  badge: "#063a4e", text: "#cffafe" },
  default:    { bg: "#111827", border: "#6b7280", accent: "#9ca3af", glow: "rgba(107,114,128,0.2)", badge: "#1f2937", text: "#e5e7eb" },
};

const ICONS = {
  patient: "🧠", disorder: "⚠️", symptom: "🔴",
  therapy: "💬", medication: "💊", lifestyle: "🌿", default: "●",
};

// ── Layout constants ─────────────────────────────────────────────────────────
const COL = { patient: 160, disorder: 440, symptom: 740, therapy: 1040, medication: 1340 };
const COL_LIFESTYLE = 740;
const SVG_W = 1580;
const TOP_PAD    = 100;
const BOTTOM_PAD = 80;
const CARD_W  = 220;
const ROOT_W  = 250;
const CARD_H  = 72;
const ROOT_H  = 88;
const ROW_GAP    = 30;
const BRANCH_GAP = 50;

// ── Helpers ──────────────────────────────────────────────────────────────────
function buildMaps(nodes, edges) {
  const nodeMap   = Object.fromEntries(nodes.map(n => [n.id, n]));
  const childMap  = {};
  const parentMap = {};
  for (const { source, target } of edges) {
    (childMap[source] ||= []).push(target);
    parentMap[target] = source;
  }
  return { nodeMap, childMap, parentMap };
}

function wrapText(text = "", maxChars = 22, maxLines = 2) {
  const words = String(text).split(/\s+/).filter(Boolean);
  if (!words.length) return [""];
  const lines = [];
  let current = "";
  for (const word of words) {
    const test = current ? `${current} ${word}` : word;
    if (test.length <= maxChars) { current = test; }
    else {
      if (current) lines.push(current);
      current = word;
      if (lines.length === maxLines - 1) break;
    }
  }
  if (lines.length < maxLines && current) lines.push(current);
  const used = lines.join(" ").split(/\s+/).filter(Boolean).length;
  if (used < words.length && lines.length)
    lines[lines.length - 1] = `${lines[lines.length - 1].slice(0, maxChars - 1)}…`;
  return lines.slice(0, maxLines);
}

function nodeSize(type) {
  return { w: type === "patient" ? ROOT_W : CARD_W, h: type === "patient" ? ROOT_H : CARD_H };
}

function curvePath(x1, y1, x2, y2) {
  const dx = Math.max(70, (x2 - x1) * 0.44);
  return `M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2-dx} ${y2}, ${x2} ${y2}`;
}

function buildLayout(nodes, edges) {
  const { nodeMap, childMap } = buildMaps(nodes, edges);
  const pos = {};

  const root = nodes.find(n => n.type === "patient") || nodes[0];
  if (!root) return { pos: {}, height: 400, headers: [] };

  const rootChildren = childMap[root.id] || [];
  const disorders  = rootChildren.filter(id => nodeMap[id]?.type === "disorder");
  const lifestyles = rootChildren.filter(id => nodeMap[id]?.type === "lifestyle");
  const extras     = rootChildren.filter(id => nodeMap[id] && !["disorder","lifestyle"].includes(nodeMap[id].type));

  const branches = disorders.map(did => {
    const dChildren  = (childMap[did] || []);
    const symptoms   = dChildren.filter(id => nodeMap[id]?.type === "symptom");
    const therapies  = dChildren.filter(id => nodeMap[id]?.type === "therapy");
    const medications = [], nestedTherapies = [];
    therapies.forEach(tid => {
      (childMap[tid] || []).forEach(cid => {
        const t = nodeMap[cid]?.type;
        if (t === "medication") medications.push(cid);
        if (t === "therapy")   nestedTherapies.push(cid);
      });
    });
    const allTherapies = [...therapies, ...nestedTherapies];
    const rows = Math.max(symptoms.length, allTherapies.length, medications.length, 1);
    return { disorder: did, symptoms, therapies: allTherapies, medications, rows };
  });

  let y = TOP_PAD;
  const disorderCenters = [];

  branches.forEach(branch => {
    const rowH       = CARD_H + ROW_GAP;
    const blockH     = Math.max(1, branch.rows) * rowH - ROW_GAP;
    const centerY    = y + blockH / 2;
    pos[branch.disorder] = { x: COL.disorder, y: centerY };
    const rowY = i => y + i * rowH + CARD_H / 2;
    branch.symptoms.forEach((id, i)    => { pos[id] = { x: COL.symptom,   y: rowY(i) }; });
    branch.therapies.forEach((id, i)   => { pos[id] = { x: COL.therapy,   y: rowY(i) }; });
    branch.medications.forEach((id, i) => { pos[id] = { x: COL.medication, y: rowY(i) }; });
    disorderCenters.push(centerY);
    y += blockH + BRANCH_GAP;
  });

  if (lifestyles.length) {
    const rowH = CARD_H + ROW_GAP;
    lifestyles.forEach((id, i) => { pos[id] = { x: COL_LIFESTYLE, y: y + i * rowH + CARD_H / 2 }; });
    y += lifestyles.length * rowH + BRANCH_GAP;
  }

  if (extras.length) {
    const rowH = CARD_H + ROW_GAP;
    extras.forEach((id, i) => { pos[id] = { x: COL.disorder, y: y + i * rowH + CARD_H / 2 }; });
    y += extras.length * rowH + BRANCH_GAP;
  }

  const patientY = disorderCenters.length
    ? (disorderCenters[0] + disorderCenters[disorderCenters.length - 1]) / 2
    : TOP_PAD + 140;
  pos[root.id] = { x: COL.patient, y: patientY };

  nodes.forEach((n, i) => {
    if (!pos[n.id]) {
      pos[n.id] = { x: 100 + (i % 4) * 280, y: y + Math.floor(i / 4) * (CARD_H + ROW_GAP) };
    }
  });

  const headers = [
    { key: "patient",    label: "Patient",    x: COL.patient },
    { key: "disorder",   label: "Disorder",   x: COL.disorder },
    { key: "symptom",    label: "Symptoms",   x: COL.symptom },
    { key: "therapy",    label: "Therapy",    x: COL.therapy },
    { key: "medication", label: "Medication", x: COL.medication },
  ].filter(({ key }) => nodes.some(n => n.type === key));

  return { pos, height: Math.max(y + BOTTOM_PAD, 480), headers };
}

function fitView(svgW, svgH, containerW, containerH) {
  const pad   = 48;
  const scale = Math.min((containerW - pad * 2) / svgW, (containerH - pad * 2) / svgH, 1.2);
  const x     = (containerW - svgW * scale) / 2;
  const y     = (containerH - svgH * scale) / 2;
  return { scale, x, y };
}

// ── Component ────────────────────────────────────────────────────────────────
export default function MindMap({ data }) {
  const svgRef      = useRef(null);
  const wrapperRef  = useRef(null);

  const [hovered, setHovered] = useState(null);
  const [view,    setView]    = useState({ scale: 1, x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef({ ox: 0, oy: 0, startX: 0, startY: 0 });

  const { nodes = [], edges = [] } = data || {};
  const { pos, height, headers }  = useMemo(
    () => (!nodes.length ? { pos: {}, height: 480, headers: [] } : buildLayout(nodes, edges)),
    [nodes, edges]
  );

  const connected = useMemo(() => {
    if (!hovered) return new Set();
    return new Set(edges.flatMap(e => {
      if (e.source === hovered) return [e.target];
      if (e.target === hovered) return [e.source];
      return [];
    }));
  }, [hovered, edges]);

  const doFit = useCallback(() => {
    const el = wrapperRef.current;
    if (!el) return;
    setView(fitView(SVG_W, height, el.clientWidth, 640));
  }, [height]);

  useEffect(() => {
    const t = setTimeout(doFit, 60);
    return () => clearTimeout(t);
  }, [doFit, nodes.length]);

  // Pan via mouse drag (no wheel zoom)
  const handlePointerDown = e => {
    if (e.target.closest("[data-no-pan='true']")) return;
    setDragging(true);
    dragRef.current = { ox: e.clientX, oy: e.clientY, startX: view.x, startY: view.y };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const handlePointerMove = e => {
    if (!dragging) return;
    setView(v => ({
      ...v,
      x: dragRef.current.startX + (e.clientX - dragRef.current.ox),
      y: dragRef.current.startY + (e.clientY - dragRef.current.oy),
    }));
  };
  const handlePointerUp = () => setDragging(false);

  const setZoom = delta => setView(v => ({
    ...v,
    scale: Math.max(0.3, Math.min(2.8, +(v.scale + delta).toFixed(3))),
  }));
  const resetView = () => doFit();

  // PNG export
  const downloadPNG = () => {
    const svg = svgRef.current;
    if (!svg) return;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width",  SVG_W);
    clone.setAttribute("height", height);
    const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml;charset=utf-8" });
    const url  = URL.createObjectURL(blob);
    const img  = new Image();
    img.onload = () => {
      const s = 2.5, c = document.createElement("canvas");
      c.width = SVG_W * s; c.height = height * s;
      const ctx = c.getContext("2d");
      ctx.scale(s, s);
      ctx.fillStyle = "#030712"; ctx.fillRect(0, 0, SVG_W, height);
      ctx.drawImage(img, 0, 0, SVG_W, height);
      URL.revokeObjectURL(url);
      Object.assign(document.createElement("a"), { href: c.toDataURL("image/png"), download: "mindmap.png" }).click();
    };
    img.onerror = () => { URL.revokeObjectURL(url); alert("Export failed."); };
    img.src = url;
  };

  // ── Render ────────────────────────────────────────────────────────────────
  if (!nodes.length) {
    return (
      <div style={styles.empty}>
        <span style={{ fontSize: 32, marginBottom: 10 }}>🧬</span>
        Analyze a consultation to generate the mind map
      </div>
    );
  }

  return (
    <div style={styles.shell}>
      {/* ── Toolbar ── */}
      <div style={styles.toolbar}>
        {/* Legend */}
        <div style={styles.legendRow}>
          {Object.entries(COLORS).filter(([k]) => k !== "default" && nodes.some(n => n.type === k))
            .map(([type, cfg]) => (
              <div key={type} style={styles.legendItem}>
                <span style={{ ...styles.legendDot, background: cfg.border, boxShadow: `0 0 8px ${cfg.glow}` }} />
                <span style={{ color: cfg.accent, fontSize: 11, fontWeight: 600, textTransform: "capitalize", letterSpacing: "0.4px" }}>
                  {ICONS[type]} {type}
                </span>
              </div>
          ))}
        </div>

        {/* Controls */}
        <div style={styles.controlRow}>
          {/* Zoom slider */}
          <div style={styles.zoomGroup}>
            <button onClick={() => setZoom(-0.1)} style={styles.iconBtn}>−</button>
            <div style={styles.zoomTrack}>
              <div style={{ ...styles.zoomFill, width: `${((view.scale - 0.3) / 2.5) * 100}%` }} />
              <input
                type="range" min={30} max={280} step={1}
                value={Math.round(view.scale * 100)}
                onChange={e => setView(v => ({ ...v, scale: +e.target.value / 100 }))}
                style={styles.zoomRange}
              />
            </div>
            <button onClick={() => setZoom(+0.1)} style={styles.iconBtn}>+</button>
            <span style={styles.zoomLabel}>{Math.round(view.scale * 100)}%</span>
          </div>

          <button onClick={resetView} style={styles.actionBtn}>⊞ Fit</button>
          <button onClick={downloadPNG} style={styles.downloadBtn}>⬇ PNG</button>
        </div>
      </div>

      {/* ── Canvas ── */}
      <div
        ref={wrapperRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{
          ...styles.canvas,
          cursor: dragging ? "grabbing" : "grab",
        }}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SVG_W} ${height}`}
          width="100%"
          height="100%"
          style={{ display: "block" }}
        >
          <defs>
            {/* Card gradients */}
            {Object.entries(COLORS).map(([type, cfg]) => (
              <linearGradient key={`grad_${type}`} id={`grad_${type}`} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%"   stopColor={cfg.bg} stopOpacity="1" />
                <stop offset="100%" stopColor={cfg.border} stopOpacity="0.18" />
              </linearGradient>
            ))}
            {/* Glow filters */}
            {Object.entries(COLORS).map(([type, cfg]) => (
              <filter key={`glow_${type}`} id={`glow_${type}`} x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur" />
                <feColorMatrix in="blur" type="saturate" values="2" result="sat" />
                <feMerge>
                  <feMergeNode in="sat" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            ))}
            {/* Subtle drop shadow */}
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="4" stdDeviation="10" floodColor="#000" floodOpacity="0.55" />
            </filter>
            {/* Dot grid */}
            <pattern id="dots" x="0" y="0" width="28" height="28" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="#ffffff" opacity="0.035" />
            </pattern>
            {/* Edge gradient */}
            <linearGradient id="edgeGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#3b82f6" stopOpacity="0.7" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.7" />
            </linearGradient>
          </defs>

          {/* Background */}
          <rect width={SVG_W} height={height} fill="#030712" />
          <rect width={SVG_W} height={height} fill="url(#dots)" />
          {/* Ambient blobs */}
          <ellipse cx={300}  cy={200}  rx={260} ry={160} fill="rgba(59,130,246,0.04)"  />
          <ellipse cx={1300} cy={500}  rx={300} ry={180} fill="rgba(139,92,246,0.04)"  />
          <ellipse cx={800}  cy={height*0.7} rx={280} ry={160} fill="rgba(16,185,129,0.04)" />

          <g transform={`translate(${view.x}, ${view.y}) scale(${view.scale})`}>

            {/* Column dividers */}
            {[COL.disorder - 60, COL.symptom - 60, COL.therapy - 60, COL.medication - 60].map(x => (
              <line key={x} x1={x} y1={55} x2={x} y2={height - 30}
                stroke="#1e3a5f" strokeWidth="1" strokeDasharray="6 9" opacity="0.5" />
            ))}

            {/* Column headers */}
            {headers.map(h => {
              const cfg = COLORS[h.key] || COLORS.default;
              return (
                <g key={h.key}>
                  <rect x={h.x - 58} y={16} width={116} height={28} rx={14}
                    fill={cfg.bg} stroke={cfg.border} strokeOpacity="0.5" />
                  <text x={h.x} y={34} textAnchor="middle"
                    fill={cfg.accent} fontSize="10" fontWeight="800" letterSpacing="1.4"
                    fontFamily="'Outfit','Segoe UI',sans-serif">
                    {h.label.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {/* Edges */}
            {edges.map((e, i) => {
              const s  = nodes.find(n => n.id === e.source);
              const t  = nodes.find(n => n.id === e.target);
              const sp = pos[e.source], tp = pos[e.target];
              if (!s || !t || !sp || !tp) return null;

              const sw  = nodeSize(s.type).w;
              const tw  = nodeSize(t.type).w;
              const active = hovered === e.source || hovered === e.target;
              const dimmed = hovered && !active;
              const cfg    = COLORS[s.type] || COLORS.default;

              return (
                <path key={i}
                  d={curvePath(sp.x + sw/2, sp.y, tp.x - tw/2, tp.y)}
                  fill="none"
                  stroke={active ? cfg.border : "#1e3a5f"}
                  strokeWidth={active ? 3 : 1.8}
                  opacity={dimmed ? 0.06 : active ? 0.95 : 0.45}
                  strokeLinecap="round"
                  strokeDasharray={active ? "none" : "none"}
                  style={{ transition: "opacity 0.18s, stroke-width 0.18s" }}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map(node => {
              const p = pos[node.id];
              if (!p) return null;

              const cfg     = COLORS[node.type] || COLORS.default;
              const isRoot  = node.type === "patient";
              const isHover = hovered === node.id;
              const isLinked = connected.has(node.id);
              const dimmed  = hovered && !isHover && !isLinked;

              const { w, h }   = nodeSize(node.type);
              const x = p.x - w/2, y = p.y - h/2;
              const rx = 18;
              const labelLines = wrapText(node.label || node.id, isRoot ? 26 : 24, 2);

              return (
                <g key={node.id} data-no-pan="true"
                  transform={`translate(${x}, ${y})`}
                  onMouseEnter={() => setHovered(node.id)}
                  onMouseLeave={() => setHovered(null)}
                  style={{ cursor: "pointer", opacity: dimmed ? 0.12 : 1, transition: "opacity 0.2s" }}>

                  {/* Outer glow ring on hover/root */}
                  {(isRoot || isHover) && (
                    <rect x={-8} y={-8} width={w+16} height={h+16} rx={rx+6}
                      fill={cfg.border} opacity={0.13}
                      filter={`url(#glow_${node.type || "default"})`} />
                  )}

                  {/* Drop shadow */}
                  <rect x={3} y={6} width={w} height={h} rx={rx}
                    fill="#000" opacity={0.5} />

                  {/* Card body */}
                  <rect width={w} height={h} rx={rx}
                    fill={`url(#grad_${node.type || "default"})`}
                    stroke={cfg.border}
                    strokeWidth={isHover || isRoot ? 2.4 : 1.4}
                    strokeOpacity={isHover || isRoot ? 1 : 0.7}
                    filter={isHover || isRoot ? `url(#shadow)` : undefined}
                  />

                  {/* Left accent bar */}
                  <rect x={0} y={0} width={5} height={h} rx={5}
                    fill={cfg.border} opacity={1} />

                  {/* Right side shimmer strip */}
                  <rect x={w-5} y={0} width={5} height={h} rx={5}
                    fill={cfg.border} opacity={0.2} />

                  {/* Icon circle */}
                  <circle cx={30} cy={h/2} r={isRoot ? 22 : 19}
                    fill={cfg.bg} stroke={cfg.border} strokeWidth="1.6" />
                  <text x={30} y={h/2 + 5} textAnchor="middle"
                    fontSize={isRoot ? 18 : 15}
                    style={{ pointerEvents: "none", userSelect: "none" }}>
                    {ICONS[node.type] || ICONS.default}
                  </text>

                  {/* Type badge */}
                  <rect x={58} y={8} width={isRoot ? 80 : 70} height={17} rx={8}
                    fill={cfg.badge} opacity={0.9} />
                  <text x={93} y={20.5} textAnchor="middle"
                    fill={cfg.accent} fontSize="8.5" fontWeight="800"
                    letterSpacing="1.1" fontFamily="'Outfit','Segoe UI',sans-serif">
                    {(node.type || "node").toUpperCase()}
                  </text>

                  {/* Label */}
                  {labelLines.map((line, idx) => (
                    <text key={idx}
                      x={60}
                      y={(isRoot ? 39 : 36) + idx * 15}
                      fill="#f1f5f9"
                      fontSize={isRoot ? 13 : 12}
                      fontWeight={isRoot ? "700" : "600"}
                      fontFamily="'Outfit','Segoe UI',sans-serif"
                      style={{ userSelect: "none" }}>
                      {line}
                    </text>
                  ))}
                </g>
              );
            })}

            {/* Hover tooltip */}
            {hovered && (() => {
              const n  = nodes.find(m => m.id === hovered);
              const p  = pos[hovered];
              if (!n || !p) return null;
              const cfg   = COLORS[n.type] || COLORS.default;
              const { w } = nodeSize(n.type);
              const tx    = Math.min(p.x + w/2 + 16, SVG_W - 240);
              const ty    = Math.max(10, p.y - 38);
              const lines = wrapText(n.label || n.id, 30, 3);

              return (
                <g data-no-pan="true" style={{ pointerEvents: "none" }}>
                  {/* Arrow */}
                  <polygon
                    points={`${tx-8},${ty+22} ${tx},${ty+16} ${tx},${ty+28}`}
                    fill="#0f172a" stroke={cfg.border} strokeWidth="1" strokeOpacity="0.8" />
                  <rect x={tx} y={ty} width={230} height={24 + lines.length * 14} rx={10}
                    fill="#0f172a" stroke={cfg.border} strokeOpacity="0.8" strokeWidth="1.4" />
                  <text x={tx+14} y={ty+15}
                    fill={cfg.accent} fontSize="8" fontWeight="800" letterSpacing="1.2"
                    fontFamily="'Outfit',sans-serif">
                    {(n.type || "node").toUpperCase()}
                  </text>
                  {lines.map((line, i) => (
                    <text key={i} x={tx+14} y={ty+29+i*14}
                      fill="#e2e8f0" fontSize="11" fontWeight="600"
                      fontFamily="'Outfit',sans-serif">
                      {line}
                    </text>
                  ))}
                </g>
              );
            })()}
          </g>
        </svg>
      </div>

      {/* ── Footer ── */}
      <div style={styles.footer}>
        <span>🖱 Drag to pan</span>
        <span style={{ opacity: 0.4 }}>·</span>
        <span>Use the zoom slider or ± buttons</span>
        <span style={{ opacity: 0.4 }}>·</span>
        <span>⊞ Fit resets view</span>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  shell: {
    background: "#030712",
    borderRadius: 18,
    border: "1px solid #1e3a5f",
    overflow: "hidden",
    boxShadow: "0 24px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04) inset",
    fontFamily: "'Outfit','Segoe UI',sans-serif",
  },
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
    padding: "12px 18px",
    borderBottom: "1px solid #0f1f38",
    background: "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))",
  },
  legendRow: {
    display: "flex",
    gap: 14,
    flexWrap: "wrap",
    alignItems: "center",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  legendDot: {
    width: 9,
    height: 9,
    borderRadius: 3,
    display: "inline-block",
    flexShrink: 0,
  },
  controlRow: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  zoomGroup: {
    display: "flex",
    alignItems: "center",
    gap: 7,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid #1e3a5f",
    borderRadius: 10,
    padding: "5px 10px",
  },
  iconBtn: {
    background: "none",
    border: "none",
    color: "#64748b",
    fontSize: 17,
    lineHeight: 1,
    cursor: "pointer",
    padding: "0 2px",
    fontWeight: 700,
    transition: "color 0.15s",
  },
  zoomTrack: {
    position: "relative",
    width: 100,
    height: 4,
    background: "#1e3a5f",
    borderRadius: 4,
    overflow: "visible",
  },
  zoomFill: {
    position: "absolute",
    left: 0, top: 0, height: "100%",
    background: "linear-gradient(90deg, #3b82f6, #8b5cf6)",
    borderRadius: 4,
    pointerEvents: "none",
    transition: "width 0.1s",
  },
  zoomRange: {
    position: "absolute",
    top: -8, left: 0,
    width: "100%",
    opacity: 0,
    cursor: "pointer",
    height: 20,
    margin: 0,
  },
  zoomLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: 700,
    minWidth: 36,
    textAlign: "right",
    fontFamily: "monospace",
  },
  actionBtn: {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid #1e3a5f",
    color: "#94a3b8",
    borderRadius: 9,
    padding: "6px 12px",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "0.3px",
  },
  downloadBtn: {
    background: "linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.15))",
    border: "1px solid #3b82f688",
    color: "#60a5fa",
    borderRadius: 9,
    padding: "6px 14px",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "0.3px",
  },
  canvas: {
    height: 640,
    overflow: "hidden",
    touchAction: "none",
    background:
      "radial-gradient(ellipse at 10% 20%, rgba(59,130,246,0.07) 0%, transparent 40%), " +
      "radial-gradient(ellipse at 90% 70%, rgba(139,92,246,0.06) 0%, transparent 40%), " +
      "#030712",
    userSelect: "none",
  },
  footer: {
    padding: "8px 18px",
    borderTop: "1px solid #0f1f38",
    fontSize: 10.5,
    color: "#334155",
    display: "flex",
    gap: 10,
    background: "rgba(0,0,0,0.3)",
    fontWeight: 500,
  },
  empty: {
    padding: 56,
    textAlign: "center",
    color: "#334155",
    fontSize: 13,
    background: "#030712",
    borderRadius: 16,
    border: "1px dashed #1e3a5f",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 8,
    fontFamily: "'Outfit','Segoe UI',sans-serif",
  },
};