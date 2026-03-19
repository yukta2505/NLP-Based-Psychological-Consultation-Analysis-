import { useEffect, useMemo, useRef, useState } from "react";

const COLORS = {
  patient: { fill: "#102447", stroke: "#4f8eff", soft: "rgba(79,142,255,0.14)" },
  disorder: { fill: "#471616", stroke: "#ff6b6b", soft: "rgba(255,107,107,0.14)" },
  symptom: { fill: "#4b310a", stroke: "#f5b041", soft: "rgba(245,176,65,0.14)" },
  therapy: { fill: "#103524", stroke: "#34d399", soft: "rgba(52,211,153,0.14)" },
  medication: { fill: "#2f1b57", stroke: "#a78bfa", soft: "rgba(167,139,250,0.14)" },
  lifestyle: { fill: "#0f3040", stroke: "#38bdf8", soft: "rgba(56,189,248,0.14)" },
  default: { fill: "#24324c", stroke: "#7c8daa", soft: "rgba(124,141,170,0.14)" },
};

const ICONS = {
  patient: "🧠",
  disorder: "⚠",
  symptom: "◉",
  therapy: "💬",
  medication: "💊",
  lifestyle: "🌿",
  default: "•",
};

const COL = {
  patient: 140,
  disorder: 390,
  symptom: 690,
  therapy: 990,
  medication: 1290,
  lifestyle: 690,
};

const SVG_W = 1500;
const TOP_PAD = 90;
const BOTTOM_PAD = 70;
const CARD_W = 210;
const ROOT_W = 240;
const CARD_H = 64;
const ROOT_H = 78;
const ROW_GAP = 22;
const BRANCH_GAP = 42;

function buildMaps(nodes, edges) {
  const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const childMap = {};
  const parentMap = {};

  for (const { source, target } of edges) {
    if (!childMap[source]) childMap[source] = [];
    childMap[source].push(target);
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
    if (test.length <= maxChars) {
      current = test;
    } else {
      if (current) lines.push(current);
      current = word;
      if (lines.length === maxLines - 1) break;
    }
  }

  if (lines.length < maxLines && current) lines.push(current);

  const usedWords = lines.join(" ").split(/\s+/).filter(Boolean).length;
  if (usedWords < words.length && lines.length) {
    lines[lines.length - 1] = `${lines[lines.length - 1].slice(0, maxChars - 2)}…`;
  }

  return lines.slice(0, maxLines);
}

function nodeSize(type) {
  return {
    w: type === "patient" ? ROOT_W : CARD_W,
    h: type === "patient" ? ROOT_H : CARD_H,
  };
}

function curvePath(x1, y1, x2, y2) {
  const dx = Math.max(60, (x2 - x1) * 0.42);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function buildLayout(nodes, edges) {
  const { nodeMap, childMap } = buildMaps(nodes, edges);
  const pos = {};

  const root = nodes.find((n) => n.type === "patient") || nodes[0];
  if (!root) return { pos: {}, height: 400, headers: [] };

  const rootChildren = childMap[root.id] || [];

  const disorders = rootChildren.filter((id) => nodeMap[id]?.type === "disorder");
  const lifestyles = rootChildren.filter((id) => nodeMap[id]?.type === "lifestyle");
  const extras = rootChildren.filter(
    (id) => nodeMap[id] && !["disorder", "lifestyle"].includes(nodeMap[id].type)
  );

  const branches = disorders.map((did) => {
    const dChildren = childMap[did] || [];
    const symptoms = dChildren.filter((id) => nodeMap[id]?.type === "symptom");
    const therapies = dChildren.filter((id) => nodeMap[id]?.type === "therapy");

    const medications = [];
    const nestedTherapies = [];

    therapies.forEach((tid) => {
      (childMap[tid] || []).forEach((cid) => {
        const t = nodeMap[cid]?.type;
        if (t === "medication") medications.push(cid);
        if (t === "therapy") nestedTherapies.push(cid);
      });
    });

    const allTherapies = [...therapies, ...nestedTherapies];
    const rows = Math.max(symptoms.length, allTherapies.length, medications.length, 1);

    return {
      disorder: did,
      symptoms,
      therapies: allTherapies,
      medications,
      rows,
    };
  });

  let y = TOP_PAD;
  const disorderCenters = [];

  branches.forEach((branch) => {
    const rowH = CARD_H + ROW_GAP;
    const blockHeight = Math.max(1, branch.rows) * rowH - ROW_GAP;
    const centerY = y + blockHeight / 2;

    pos[branch.disorder] = { x: COL.disorder, y: centerY };

    const rowY = (i) => y + i * rowH + CARD_H / 2;

    branch.symptoms.forEach((id, i) => {
      pos[id] = { x: COL.symptom, y: rowY(i) };
    });

    branch.therapies.forEach((id, i) => {
      pos[id] = { x: COL.therapy, y: rowY(i) };
    });

    branch.medications.forEach((id, i) => {
      pos[id] = { x: COL.medication, y: rowY(i) };
    });

    disorderCenters.push(centerY);
    y += blockHeight + BRANCH_GAP;
  });

  if (lifestyles.length) {
    const rowH = CARD_H + ROW_GAP;
    lifestyles.forEach((id, i) => {
      pos[id] = { x: COL.lifestyle, y: y + i * rowH + CARD_H / 2 };
    });
    y += lifestyles.length * rowH + BRANCH_GAP;
  }

  if (extras.length) {
    const rowH = CARD_H + ROW_GAP;
    extras.forEach((id, i) => {
      pos[id] = { x: COL.disorder, y: y + i * rowH + CARD_H / 2 };
    });
    y += extras.length * rowH + BRANCH_GAP;
  }

  const patientY =
    disorderCenters.length > 0
      ? (disorderCenters[0] + disorderCenters[disorderCenters.length - 1]) / 2
      : TOP_PAD + 120;

  pos[root.id] = { x: COL.patient, y: patientY };

  nodes.forEach((n, i) => {
    if (!pos[n.id]) {
      pos[n.id] = {
        x: 100 + (i % 4) * 260,
        y: y + Math.floor(i / 4) * (CARD_H + ROW_GAP),
      };
    }
  });

  const headers = [
    { key: "patient", label: "Patient", x: COL.patient },
    { key: "disorder", label: "Disorder", x: COL.disorder },
    { key: "symptom", label: "Symptoms", x: COL.symptom },
    { key: "therapy", label: "Therapy", x: COL.therapy },
    { key: "medication", label: "Medication", x: COL.medication },
  ].filter(({ key }) => nodes.some((n) => n.type === key || (key === "symptom" && n.type === "symptom")));

  return {
    pos,
    height: Math.max(y + BOTTOM_PAD, 420),
    headers,
  };
}

function fitView(width, height, containerW, containerH) {
  const pad = 40;
  const scale = Math.min(
    (containerW - pad * 2) / width,
    (containerH - pad * 2) / height,
    1
  );
  const x = (containerW - width * scale) / 2;
  const y = (containerH - height * scale) / 2;
  return { scale, x, y };
}

export default function MindMap({ data }) {
  const svgRef = useRef(null);
  const wrapperRef = useRef(null);

  const [hovered, setHovered] = useState(null);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef({ x: 0, y: 0, startX: 0, startY: 0 });

  const { nodes = [], edges = [] } = data || {};

  const { pos, height, headers } = useMemo(() => {
    if (!nodes.length) return { pos: {}, height: 420, headers: [] };
    return buildLayout(nodes, edges);
  }, [nodes, edges]);

  const connected = useMemo(() => {
    if (!hovered) return new Set();
    return new Set(
      edges.flatMap((e) => {
        if (e.source === hovered) return [e.target];
        if (e.target === hovered) return [e.source];
        return [];
      })
    );
  }, [hovered, edges]);

  const doFit = () => {
    const el = wrapperRef.current;
    if (!el) return;
    const next = fitView(SVG_W, height, el.clientWidth, 620);
    setView(next);
  };

  useEffect(() => {
    const t = setTimeout(() => doFit(), 50);
    return () => clearTimeout(t);
  }, [height, nodes.length]);

  const zoom = (factor) => {
    setView((v) => ({
      ...v,
      scale: Math.max(0.55, Math.min(2.4, +(v.scale * factor).toFixed(3))),
    }));
  };

  const resetView = () => setView({ scale: 1, x: 0, y: 0 });

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    zoom(factor);
  };

  const handlePointerDown = (e) => {
    if (e.target.closest("[data-no-pan='true']")) return;
    setDragging(true);
    dragRef.current = {
      x: e.clientX,
      y: e.clientY,
      startX: view.x,
      startY: view.y,
    };
  };

  const handlePointerMove = (e) => {
    if (!dragging) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    setView((v) => ({
      ...v,
      x: dragRef.current.startX + dx,
      y: dragRef.current.startY + dy,
    }));
  };

  const handlePointerUp = () => setDragging(false);

  const downloadPNG = () => {
    const svg = svgRef.current;
    if (!svg) return;

    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    clone.setAttribute("width", SVG_W);
    clone.setAttribute("height", height);

    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(clone);

    if (!source.match(/^<svg[^>]+xmlns=/)) {
      source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }

    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      const scale = 2.5;
      const canvas = document.createElement("canvas");
      canvas.width = SVG_W * scale;
      canvas.height = height * scale;

      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.fillStyle = "#07090e";
      ctx.fillRect(0, 0, SVG_W, height);
      ctx.drawImage(img, 0, 0, SVG_W, height);

      URL.revokeObjectURL(url);

      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = "mindmap-full.png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      alert("PNG export failed.");
    };

    img.src = url;
  };

  if (!nodes.length) {
    return (
      <div
        style={{
          padding: 48,
          textAlign: "center",
          color: "#52617d",
          fontSize: 13,
          background: "#08090e",
          borderRadius: 14,
          border: "1px dashed #1f2e49",
        }}
      >
        Analyze a consultation to generate the mind map
      </div>
    );
  }

  return (
    <div
      style={{
        background: "#07090e",
        borderRadius: 16,
        border: "1px solid #1a2744",
        overflow: "hidden",
        boxShadow: "0 14px 40px rgba(0,0,0,0.3)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          padding: "12px 16px",
          borderBottom: "1px solid #13203a",
          background: "linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01))",
        }}
      >
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {Object.entries(COLORS)
            .filter(([k]) => k !== "default")
            .map(([type, cfg]) => {
              if (!nodes.some((n) => n.type === type)) return null;
              return (
                <div
                  key={type}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    fontSize: 11,
                    color: "#91a0be",
                    textTransform: "capitalize",
                  }}
                >
                  <span
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 3,
                      background: cfg.stroke,
                      boxShadow: `0 0 12px ${cfg.stroke}55`,
                      display: "inline-block",
                    }}
                  />
                  {type}
                </div>
              );
            })}
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button onClick={() => zoom(1.1)} style={toolBtn}>＋</button>
          <button onClick={() => zoom(0.9)} style={toolBtn}>－</button>
          <button onClick={resetView} style={toolBtn}>Reset</button>
          <button onClick={doFit} style={toolBtn}>Fit</button>
          <button onClick={downloadPNG} style={downloadBtn}>⬇ PNG</button>
        </div>
      </div>

      <div
        ref={wrapperRef}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{
          height: 620,
          overflow: "hidden",
          cursor: dragging ? "grabbing" : "grab",
          touchAction: "none",
          background:
            "radial-gradient(circle at top left, rgba(79,142,255,0.06), transparent 28%), radial-gradient(circle at top right, rgba(52,211,153,0.04), transparent 24%), #07090e",
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
            {Object.entries(COLORS).map(([type, cfg]) => (
              <linearGradient key={type} id={`card_${type}`} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={cfg.fill} stopOpacity="1" />
                <stop offset="100%" stopColor={cfg.stroke} stopOpacity="0.2" />
              </linearGradient>
            ))}

            <filter id="cardGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <pattern id="bgDots" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="0.9" fill="#fff" opacity="0.03" />
            </pattern>
          </defs>

          <rect width={SVG_W} height={height} fill="#07090e" />
          <rect width={SVG_W} height={height} fill="url(#bgDots)" />

          <g transform={`translate(${view.x}, ${view.y}) scale(${view.scale})`}>
            {[COL.disorder - 55, COL.symptom - 55, COL.therapy - 55, COL.medication - 55].map((x) => (
              <line
                key={x}
                x1={x}
                y1={50}
                x2={x}
                y2={height - 24}
                stroke="#14223d"
                strokeWidth="1"
                strokeDasharray="7 8"
              />
            ))}

            {headers.map((h) => (
              <g key={h.key}>
                <rect
                  x={h.x - 52}
                  y={18}
                  width={104}
                  height={24}
                  rx={12}
                  fill="#0f1728"
                  stroke="#1d2a45"
                />
                <text
                  x={h.x}
                  y={34}
                  textAnchor="middle"
                  fill="#7486aa"
                  fontSize="10.5"
                  fontWeight="700"
                  letterSpacing="1.2"
                  fontFamily="'Outfit', 'Segoe UI', sans-serif"
                >
                  {h.label.toUpperCase()}
                </text>
              </g>
            ))}

            {edges.map((e, i) => {
              const s = nodes.find((n) => n.id === e.source);
              const t = nodes.find((n) => n.id === e.target);
              const sp = pos[e.source];
              const tp = pos[e.target];
              if (!s || !t || !sp || !tp) return null;

              const sw = nodeSize(s.type).w;
              const tw = nodeSize(t.type).w;
              const active = hovered === e.source || hovered === e.target;
              const dimmed = hovered && !active;
              const stroke = (COLORS[s.type] || COLORS.default).stroke;

              return (
                <path
                  key={i}
                  d={curvePath(sp.x + sw / 2, sp.y, tp.x - tw / 2, tp.y)}
                  fill="none"
                  stroke={active ? stroke : "#2a3a58"}
                  strokeWidth={active ? 2.6 : 1.5}
                  opacity={dimmed ? 0.08 : active ? 1 : 0.52}
                  strokeLinecap="round"
                />
              );
            })}

            {nodes.map((node) => {
              const p = pos[node.id];
              if (!p) return null;

              const cfg = COLORS[node.type] || COLORS.default;
              const isRoot = node.type === "patient";
              const isHover = hovered === node.id;
              const isLinked = connected.has(node.id);
              const dimmed = hovered && !isHover && !isLinked;

              const { w, h } = nodeSize(node.type);
              const x = p.x - w / 2;
              const y = p.y - h / 2;
              const rx = 16;
              const labelLines = wrapText(node.label || node.id, isRoot ? 24 : 22, isRoot ? 2 : 2);

              return (
                <g
                  key={node.id}
                  data-no-pan="true"
                  transform={`translate(${x}, ${y})`}
                  onMouseEnter={() => setHovered(node.id)}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    cursor: "pointer",
                    opacity: dimmed ? 0.14 : 1,
                    transition: "opacity 0.18s ease",
                  }}
                >
                  {(isRoot || isHover) && (
                    <rect
                      x={-6}
                      y={-6}
                      width={w + 12}
                      height={h + 12}
                      rx={rx + 4}
                      fill={cfg.stroke}
                      opacity={0.12}
                    />
                  )}

                  <rect
                    x={4}
                    y={5}
                    width={w}
                    height={h}
                    rx={rx}
                    fill="#000"
                    opacity={0.32}
                  />

                  <rect
                    width={w}
                    height={h}
                    rx={rx}
                    fill={`url(#card_${node.type || "default"})`}
                    stroke={cfg.stroke}
                    strokeWidth={isRoot ? 2.2 : isHover ? 2 : 1.2}
                    strokeOpacity={0.95}
                    filter={isRoot || isHover ? "url(#cardGlow)" : undefined}
                  />

                  <rect
                    x={0}
                    y={0}
                    width={w}
                    height={h}
                    rx={rx}
                    fill={cfg.soft}
                    opacity={0.45}
                  />

                  <rect
                    x={0}
                    y={0}
                    width={6}
                    height={h}
                    rx={6}
                    fill={cfg.stroke}
                    opacity={0.95}
                  />

                  <circle
                    cx={28}
                    cy={h / 2}
                    r={isRoot ? 20 : 17}
                    fill="#0c1322"
                    stroke={cfg.stroke}
                    strokeWidth="1.4"
                  />

                  <text
                    x={28}
                    y={h / 2 + 5}
                    textAnchor="middle"
                    fontSize={isRoot ? 18 : 15}
                    style={{ pointerEvents: "none", userSelect: "none" }}
                  >
                    {ICONS[node.type] || ICONS.default}
                  </text>

                  <text
                    x={56}
                    y={isRoot ? 28 : 25}
                    fill={cfg.stroke}
                    fontSize={8.5}
                    fontWeight="700"
                    letterSpacing="1"
                    fontFamily="'Outfit', 'Segoe UI', sans-serif"
                  >
                    {(node.type || "node").toUpperCase()}
                  </text>

                  {labelLines.map((line, idx) => (
                    <text
                      key={idx}
                      x={56}
                      y={(isRoot ? 47 : 43) + idx * 14}
                      fill="#eef4ff"
                      fontSize={isRoot ? 12 : 11}
                      fontWeight={isRoot ? "700" : "600"}
                      fontFamily="'Outfit', 'Segoe UI', sans-serif"
                    >
                      {line}
                    </text>
                  ))}
                </g>
              );
            })}

            {hovered &&
              (() => {
                const n = nodes.find((m) => m.id === hovered);
                const p = pos[hovered];
                if (!n || !p) return null;

                const cfg = COLORS[n.type] || COLORS.default;
                const tx = Math.min(p.x + 120, SVG_W - 250);
                const ty = Math.max(8, p.y - 40);
                const lines = wrapText(n.label || n.id, 28, 2);

                return (
                  <g data-no-pan="true">
                    <rect
                      x={tx}
                      y={ty}
                      width={220}
                      height={48}
                      rx={10}
                      fill="#0d1525"
                      stroke={cfg.stroke}
                      strokeOpacity="0.8"
                    />
                    <text
                      x={tx + 12}
                      y={ty + 16}
                      fill={cfg.stroke}
                      fontSize="8.5"
                      fontWeight="700"
                      letterSpacing="0.9"
                      fontFamily="'Outfit', sans-serif"
                    >
                      {(n.type || "node").toUpperCase()}
                    </text>
                    {lines.map((line, i) => (
                      <text
                        key={i}
                        x={tx + 12}
                        y={ty + 31 + i * 12}
                        fill="#e8eefb"
                        fontSize="10.5"
                        fontWeight="600"
                        fontFamily="'Outfit', sans-serif"
                      >
                        {line}
                      </text>
                    ))}
                  </g>
                );
              })()}
          </g>
        </svg>
      </div>

      <div
        style={{
          padding: "9px 16px",
          borderTop: "1px solid #111b2e",
          fontSize: 10.5,
          color: "#3e5478",
          textAlign: "right",
          background: "rgba(0,0,0,0.24)",
        }}
      >
        Drag to pan · Scroll to zoom · Fit to screen · Full PNG export
      </div>
    </div>
  );
}

const toolBtn = {
  background: "rgba(255,255,255,0.04)",
  border: "1px solid #223352",
  color: "#c7d5f0",
  borderRadius: 9,
  padding: "6px 11px",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 600,
};

const downloadBtn = {
  background: "rgba(79,142,255,0.14)",
  border: "1px solid #4f8eff55",
  color: "#4f8eff",
  borderRadius: 9,
  padding: "6px 12px",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 700,
};