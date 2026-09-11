const canvas = document.getElementById("vectorCanvas");
const ctx = canvas.getContext("2d");
const inspector = document.getElementById("vectorInspector");
const clusterCard = document.getElementById("clusterCard");
const statsEl = document.getElementById("vectorStats");
const tooltip = document.getElementById("vectorTooltip");
const legendEl = document.getElementById("vectorLegend");

const controls = {
  search: document.getElementById("vectorSearch"),
  relationship: document.getElementById("relationshipFilter"),
  intent: document.getElementById("intentFilter"),
  source: document.getElementById("sourceFilter"),
  memoryType: document.getElementById("memoryTypeFilter"),
  motion: document.getElementById("motionMode"),
  colorMode: document.getElementById("colorMode"),
  spread: document.getElementById("spreadSlider"),
  synthetic: document.getElementById("syntheticToggle"),
  neighbours: document.getElementById("neighbourToggle"),
  autoRotate: document.getElementById("autoRotateToggle"),
  fit: document.getElementById("fitDataButton"),
  reset: document.getElementById("resetCameraButton"),
  fullscreen: document.getElementById("fullscreenButton"),
};

const palettes = {
  source: {
    correction: "#27f5ff",
    real: "#ffcc66",
    synthetic: "#8f7bff",
    thread_episode: "#5eead4",
  },
  memory: {
    reply_example: "#93c5fd",
    thread_episode: "#5eead4",
  },
  authority: {
    high: "#2fffea",
    medium: "#f5d267",
    low: "#7b8dff",
  },
  relationship: {
    close_friend: "#27f5ff",
    casual_friend: "#ffcc66",
    family: "#7cf29b",
    university: "#ff7ab8",
    professional: "#8fb7ff",
    unknown: "#b7c0cf",
    romantic_interest: "#ff4d75",
  },
  intent: {
    greeting: "#27f5ff",
    planning: "#ffcc66",
    simple_question: "#8fb7ff",
    casual_checkin: "#7cf29b",
    emotional: "#ff7ab8",
    joke_banter_safe: "#c084fc",
    argument: "#ff6b6b",
    romantic_flirty: "#ff4d75",
    urgent: "#ff8a3d",
    professional: "#6ea8ff",
    apology: "#a5b4fc",
    thanks: "#facc15",
    confirmation: "#34d399",
    decline: "#fb7185",
    delay: "#f59e0b",
    studying_work: "#38bdf8",
    availability: "#22c55e",
    follow_up: "#e879f9",
    unknown: "#cbd5e1",
  },
};

const state = {
  points: [],
  clusters: [],
  projected: [],
  dust: [],
  selected: null,
  hover: null,
  neighbours: [],
  clusterSummary: null,
  rotationX: -0.28,
  rotationY: 0.4,
  targetRotationX: -0.28,
  targetRotationY: 0.4,
  zoom: 2.9,
  targetZoom: 2.9,
  panX: 80,
  panY: 42,
  targetPanX: 80,
  targetPanY: 42,
  dragging: false,
  lastX: 0,
  lastY: 0,
  source: "...",
  projection: "...",
  bounds: { radius: 250 },
  startedAt: performance.now(),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  createDust();
}

function createDust() {
  const count = Math.min(180, Math.max(70, Math.floor((window.innerWidth * window.innerHeight) / 9500)));
  state.dust = Array.from({ length: count }, (_, index) => {
    const seed = Math.sin(index * 999.91) * 10000;
    const frac = seed - Math.floor(seed);
    return {
      x: (Math.sin(index * 31.7) * 0.5 + 0.5) * window.innerWidth,
      y: (Math.cos(index * 17.3) * 0.5 + 0.5) * window.innerHeight,
      r: 0.4 + frac * 1.8,
      a: 0.12 + frac * 0.28,
      p: index * 0.47,
    };
  });
}

function filterPoint(point) {
  const query = controls.search.value.trim().toLowerCase();
  if (!controls.synthetic.checked && point.point_type === "synthetic") return false;
  if (controls.relationship.value !== "all" && point.relationship_type !== controls.relationship.value) return false;
  if (controls.intent.value !== "all" && point.intent_type !== controls.intent.value) return false;
  if (controls.memoryType.value !== "all" && point.memory_type !== controls.memoryType.value) return false;
  if (controls.source.value !== "all" && point.point_type !== controls.source.value && point.memory_type !== controls.source.value) return false;
  if (!query) return true;
  return [point.incoming, point.my_reply, point.summary, point.contact_name, point.relationship_type, point.intent_type, point.source, point.memory_type, point.cluster_key]
    .join(" ")
    .toLowerCase()
    .includes(query);
}

function normalisePoints(points) {
  if (!points.length) return [];
  const bounds = points.reduce((acc, point) => ({
    minX: Math.min(acc.minX, point.x),
    maxX: Math.max(acc.maxX, point.x),
    minY: Math.min(acc.minY, point.y),
    maxY: Math.max(acc.maxY, point.y),
    minZ: Math.min(acc.minZ, point.z),
    maxZ: Math.max(acc.maxZ, point.z),
  }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity });
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const cz = (bounds.minZ + bounds.maxZ) / 2;
  const radius = Math.max(...points.map((point) => Math.hypot(point.x - cx, point.y - cy, point.z - cz))) || 1;
  state.bounds = { radius };
  return points.map((point) => ({
    ...point,
    nx: (point.x - cx) / radius,
    ny: (point.y - cy) / radius,
    nz: (point.z - cz) / radius,
  }));
}

function displayPosition(point) {
  const spread = Number(controls.spread.value || 1.55);
  const visualRadius = Math.min(window.innerWidth, window.innerHeight) * 0.48 * spread;
  return {
    x: point.nx * visualRadius,
    y: point.ny * visualRadius,
    z: point.nz * visualRadius,
  };
}

function rotate(point, time) {
  const drift = controls.motion.value === "still" ? 0 : Math.sin(time * 0.0007 + point.phase) * point.drift;
  const pos = displayPosition(point);
  const x = pos.x + Math.sin(time * 0.00045 + point.phase) * drift;
  const y = pos.y + Math.cos(time * 0.0005 + point.phase) * drift;
  const z = pos.z + Math.sin(time * 0.00038 + point.phase * 0.7) * drift;
  const sinY = Math.sin(state.rotationY);
  const cosY = Math.cos(state.rotationY);
  const sinX = Math.sin(state.rotationX);
  const cosX = Math.cos(state.rotationX);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;
  const depth = 900 + z2 * state.zoom;
  const scale = Math.max(0.22, 900 / Math.max(140, depth));
  return {
    ...point,
    sx: window.innerWidth / 2 + x1 * state.zoom * scale + state.panX,
    sy: window.innerHeight / 2 + y1 * state.zoom * scale + state.panY,
    depth,
    scale,
    wx: x,
    wy: y,
    wz: z,
  };
}

function colorFor(point) {
  const mode = controls.colorMode.value;
  if (mode === "source") return point.memory_type === "thread_episode" ? palettes.source.thread_episode : palettes.source[point.point_type] || palettes.source.real;
  if (mode === "memory") return palettes.memory[point.memory_type] || palettes.memory.reply_example;
  if (mode === "authority") return palettes.authority[point.style_authority] || palettes.authority.medium;
  if (mode === "relationship") return palettes.relationship[point.relationship_type] || palettes.relationship.unknown;
  if (mode === "cluster") return hashColor(point.cluster_key || point.intent_type || "unknown");
  return palettes.intent[point.intent_type] || palettes.intent.unknown;
}

function hashColor(value) {
  let hash = 0;
  for (const char of String(value)) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 92% 68%)`;
}

function nodeImportance(point) {
  if (point.memory_type === "thread_episode") return 1.35;
  if (point.point_type === "correction") return 1.55;
  if (point.style_authority === "high") return 1.45;
  if (point.point_type === "synthetic") return 0.72;
  return 1;
}

function pointRadius(point) {
  const selected = state.selected?.id === point.id;
  const hovered = state.hover?.id === point.id;
  const neighbour = state.neighbours.some((item) => item.id === point.id);
  const base = 5.8 * nodeImportance(point);
  return base * point.scale * (selected ? 2.25 : hovered ? 1.85 : neighbour ? 1.45 : 1);
}

function drawBackground(time) {
  const gradient = ctx.createRadialGradient(window.innerWidth * 0.52, window.innerHeight * 0.42, 20, window.innerWidth * 0.5, window.innerHeight * 0.5, window.innerWidth * 0.8);
  gradient.addColorStop(0, "#14314b");
  gradient.addColorStop(0.42, "#071426");
  gradient.addColorStop(1, "#03050b");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  const nebula = ctx.createRadialGradient(window.innerWidth * 0.62, window.innerHeight * 0.58, 0, window.innerWidth * 0.62, window.innerHeight * 0.58, window.innerWidth * 0.55);
  nebula.addColorStop(0, "rgba(39,245,255,0.12)");
  nebula.addColorStop(0.35, "rgba(143,123,255,0.08)");
  nebula.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = nebula;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  ctx.save();
  for (const dust of state.dust) {
    const twinkle = Math.sin(time * 0.001 + dust.p) * 0.5 + 0.5;
    ctx.globalAlpha = dust.a * (0.45 + twinkle * 0.55);
    ctx.fillStyle = "#a7f3ff";
    ctx.beginPath();
    ctx.arc(dust.x, dust.y, dust.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawFieldRings(time) {
  ctx.save();
  ctx.translate(window.innerWidth / 2 + state.panX, window.innerHeight / 2 + state.panY + 225);
  ctx.rotate(Math.sin(time * 0.00018) * 0.035);
  for (let radius = 140; radius <= Math.max(window.innerWidth, window.innerHeight); radius += 90) {
    ctx.strokeStyle = `rgba(39, 245, 255, ${0.11 - Math.min(radius / 9000, 0.07)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(0, 0, radius, radius * 0.25, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawClusterLabels(points) {
  const summaries = new Map();
  for (const point of points) {
    const current = summaries.get(point.cluster_key);
    if (!current) summaries.set(point.cluster_key, { ...point, count: 1 });
    else {
      current.sx += point.sx;
      current.sy += point.sy;
      current.depth += point.depth;
      current.count += 1;
    }
  }
  ctx.save();
  ctx.font = "700 12px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  for (const item of [...summaries.values()].filter((entry) => entry.count >= 12).slice(0, 16)) {
    const x = item.sx / item.count;
    const y = item.sy / item.count;
    ctx.fillStyle = "rgba(206, 246, 255, 0.72)";
    ctx.shadowColor = "rgba(39,245,255,0.45)";
    ctx.shadowBlur = 10;
    ctx.fillText(`${item.intent_type} (${item.count})`, x, y - 28);
  }
  ctx.restore();
}

function drawConnections(points) {
  ctx.save();
  const selectedIds = new Set(state.neighbours.map((point) => point.id));
  if (state.selected && controls.neighbours.checked) {
    for (const point of points) {
      if (!selectedIds.has(point.id)) continue;
      ctx.strokeStyle = "rgba(39, 245, 255, 0.36)";
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(state.selected.sx, state.selected.sy);
      ctx.lineTo(point.sx, point.sy);
      ctx.stroke();
    }
  }

  const buckets = new Map();
  for (const point of points) {
    const bucket = buckets.get(point.cluster_key) || [];
    if (bucket.length < 7) bucket.push(point);
    buckets.set(point.cluster_key, bucket);
  }
  ctx.lineWidth = 0.6;
  for (const bucket of buckets.values()) {
    for (let i = 1; i < bucket.length; i += 1) {
      ctx.strokeStyle = "rgba(148, 210, 255, 0.075)";
      ctx.beginPath();
      ctx.moveTo(bucket[i - 1].sx, bucket[i - 1].sy);
      ctx.lineTo(bucket[i].sx, bucket[i].sy);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawPoint(point, time) {
  const radius = pointRadius(point);
  const color = colorFor(point);
  const selected = state.selected?.id === point.id;
  const hovered = state.hover?.id === point.id;
  const neighbour = state.neighbours.some((item) => item.id === point.id);
  const dim = state.selected && !selected && !neighbour ? 0.34 : 1;
  const pulse = 1 + Math.sin(time * 0.003 + point.phase) * (selected ? 0.12 : 0.045);
  const glowRadius = radius * (point.point_type === "synthetic" ? 3.4 : 4.9) * pulse;

  ctx.save();
  ctx.globalAlpha = (point.point_type === "synthetic" ? 0.52 : 0.96) * dim;
  ctx.shadowColor = color;
  ctx.shadowBlur = selected || hovered ? 36 : point.point_type === "correction" ? 27 : 16;
  const glow = ctx.createRadialGradient(point.sx, point.sy, 0, point.sx, point.sy, glowRadius);
  glow.addColorStop(0, "#ffffff");
  glow.addColorStop(0.18, color);
  glow.addColorStop(0.56, `${color}66`);
  glow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(point.sx, point.sy, glowRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = dim;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.sx, point.sy, radius * pulse, 0, Math.PI * 2);
  ctx.fill();

  if (selected || hovered || point.point_type === "correction") {
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.lineWidth = selected ? 2.5 : 1.2;
    ctx.beginPath();
    ctx.arc(point.sx, point.sy, radius * pulse + (selected ? 9 : 4), 0, Math.PI * 2);
    ctx.stroke();
  }
  if (point.memory_type === "thread_episode") {
    ctx.strokeStyle = selected ? "#ffffff" : "rgba(94,234,212,0.86)";
    ctx.lineWidth = selected ? 2.4 : 1.5;
    ctx.beginPath();
    ctx.rect(point.sx - radius * 1.15, point.sy - radius * 1.15, radius * 2.3, radius * 2.3);
    ctx.stroke();
  }
  ctx.restore();
}

function updateCamera(time) {
  if (controls.autoRotate.checked && !state.dragging) {
    if (controls.motion.value === "cluster_orbit") state.targetRotationY += 0.0032;
    else if (controls.motion.value === "float" || controls.motion.value === "drift") state.targetRotationY += 0.0013;
  }
  state.rotationX += (state.targetRotationX - state.rotationX) * 0.08;
  state.rotationY += (state.targetRotationY - state.rotationY) * 0.08;
  state.zoom += (state.targetZoom - state.zoom) * 0.08;
  state.panX += (state.targetPanX - state.panX) * 0.08;
  state.panY += (state.targetPanY - state.panY) * 0.08;
  if (controls.motion.value === "float") {
    state.targetPanY = 42 + Math.sin(time * 0.0008) * 12;
  }
}

function render(time) {
  updateCamera(time);
  drawBackground(time);
  drawFieldRings(time);
  const visible = state.points.filter(filterPoint).map((point) => rotate(point, time)).sort((a, b) => b.depth - a.depth);
  state.projected = visible;
  if (state.selected) {
    const selectedProjected = visible.find((point) => point.id === state.selected.id);
    if (selectedProjected) state.selected = selectedProjected;
    state.neighbours = controls.neighbours.checked && state.selected ? nearestNeighbours(state.selected, visible, 8) : [];
  }
  drawConnections(visible);
  for (const point of visible) drawPoint(point, time);
  drawClusterLabels(visible);
  updateStats(visible.length);
  requestAnimationFrame(render);
}

function nearestNeighbours(point, candidates, limit) {
  return candidates
    .filter((candidate) => candidate.id !== point.id)
    .map((candidate) => ({
      ...candidate,
      distance: Math.hypot(candidate.wx - point.wx, candidate.wy - point.wy, candidate.wz - point.wz),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
}

function updateSelectOptions() {
  const relationships = [...new Set(state.points.map((point) => point.relationship_type || "unknown"))].sort();
  const intents = [...new Set(state.points.map((point) => point.intent_type || "unknown"))].sort();
  controls.relationship.innerHTML = '<option value="all">All</option>' + relationships.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join("");
  controls.intent.innerHTML = '<option value="all">All</option>' + intents.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join("");
}

function updateLegend() {
  const mode = controls.colorMode.value;
  const entries = mode === "source"
    ? Object.entries(palettes.source)
    : mode === "memory"
      ? Object.entries(palettes.memory)
    : mode === "authority"
      ? Object.entries(palettes.authority)
      : mode === "relationship"
        ? Object.entries(palettes.relationship)
        : mode === "intent"
          ? Object.entries(palettes.intent).filter(([key]) => state.points.some((point) => point.intent_type === key)).slice(0, 10)
          : [["cluster", "#27f5ff"], ["varies", "#ffcc66"], ["by metadata", "#8f7bff"]];
  legendEl.innerHTML = entries.map(([label, color]) => `<span><i class="legend-dot" style="background:${color};color:${color}"></i>${escapeHtml(label)}</span>`).join("");
}

function updateStats(visible) {
  statsEl.innerHTML = `
    <div><span>Points</span><strong>${state.points.length}</strong></div>
    <div><span>Visible</span><strong>${visible}</strong></div>
    <div><span>Clusters</span><strong>${state.clusters.length}</strong></div>
    <div><span>Source</span><strong>${escapeHtml(state.source)}</strong></div>
    <div><span>Projection</span><strong>${escapeHtml(state.projection)}</strong></div>
    <div><span>Memory</span><strong>${escapeHtml(controls.memoryType.value)}</strong></div>
    <div><span>Mode</span><strong>${escapeHtml(controls.colorMode.value)}</strong></div>
  `;
}

function selectPoint(point) {
  state.selected = point;
  state.neighbours = controls.neighbours.checked ? nearestNeighbours(point, state.projected, 8) : [];
  const cluster = state.clusters.find((item) => item.key === point.cluster_key);
  inspector.innerHTML = `
    <p class="eyebrow">${escapeHtml(point.memory_type || "reply_example")} | ${escapeHtml(point.point_type || "point")} | ${escapeHtml(point.style_authority || "")}</p>
    <h2>${escapeHtml(point.intent_type || "unknown")}</h2>
    <div class="vector-detail">
      <span>Incoming</span>
      <p>${escapeHtml(point.incoming || "")}</p>
    </div>
    <div class="vector-detail">
      <span>Reply</span>
      <p>${escapeHtml(point.my_reply || "")}</p>
    </div>
    ${point.summary ? `<div class="vector-detail"><span>Summary</span><p>${escapeHtml(point.summary)}</p></div>` : ""}
    <div class="status-grid">
      <div><span>Relationship</span><strong>${escapeHtml(point.relationship_type || "unknown")}</strong></div>
      <div><span>Contact</span><strong>${escapeHtml(point.contact_name || "none")}</strong></div>
      <div><span>Source</span><strong>${escapeHtml(point.source || "")}</strong></div>
      <div><span>Memory</span><strong>${escapeHtml(point.memory_type || "reply_example")}</strong></div>
      <div><span>Neighbours</span><strong>${state.neighbours.length}</strong></div>
    </div>
  `;
  clusterCard.innerHTML = `
    <p class="eyebrow">Cluster</p>
    <h2>${escapeHtml(cluster?.label || point.cluster_key || "unknown")}</h2>
    <div class="status-grid">
      <div><span>Examples</span><strong>${cluster?.count || 1}</strong></div>
      <div><span>Corrections</span><strong>${cluster?.corrections || 0}</strong></div>
      <div><span>Synthetic</span><strong>${cluster?.synthetic || 0}</strong></div>
      <div><span>Relationship</span><strong>${escapeHtml(point.relationship_type || "unknown")}</strong></div>
    </div>
  `;
  flyTo(point);
}

function nearestPoint(x, y) {
  let best = null;
  let bestDistance = 26;
  for (const point of state.projected) {
    const distance = Math.hypot(point.sx - x, point.sy - y);
    if (distance < bestDistance) {
      best = point;
      bestDistance = distance;
    }
  }
  return best;
}

function updateTooltip(event) {
  if (!state.hover) {
    tooltip.classList.add("hidden");
    return;
  }
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${Math.min(window.innerWidth - 320, event.clientX + 18)}px`;
  tooltip.style.top = `${Math.min(window.innerHeight - 180, event.clientY + 18)}px`;
  tooltip.innerHTML = `
    <strong>${escapeHtml(state.hover.intent_type || "unknown")}</strong>
    <span>${escapeHtml(state.hover.relationship_type || "unknown")} | ${escapeHtml(state.hover.memory_type || "reply_example")} | ${escapeHtml(state.hover.point_type || "")}</span>
    <p>${escapeHtml(state.hover.summary || state.hover.incoming || "")}</p>
    <p>${escapeHtml(state.hover.my_reply || "")}</p>
  `;
}

function fitToData() {
  state.targetZoom = Math.max(1.65, Math.min(4.6, 850 / Math.max(window.innerWidth, window.innerHeight)));
  state.targetPanX = window.innerWidth > 1100 ? 120 : 0;
  state.targetPanY = window.innerWidth > 900 ? 46 : 120;
}

function resetCamera() {
  state.targetRotationX = -0.28;
  state.targetRotationY = 0.4;
  fitToData();
}

function flyTo(point) {
  state.targetRotationY += 0.18;
  state.targetZoom = Math.min(5.2, state.targetZoom + 0.35);
  state.targetPanX += (window.innerWidth / 2 - point.sx) * 0.12;
  state.targetPanY += (window.innerHeight / 2 - point.sy) * 0.12;
}

async function loadVectorMap() {
  const memoryType = encodeURIComponent(controls.memoryType.value || "all");
  const response = await fetch(`/api/ai-core/vector-map?limit=1800&memory_type=${memoryType}`);
  if (!response.ok) throw new Error(`Vector map failed: ${response.status}`);
  const payload = await response.json();
  const rawPoints = payload.points || [];
  state.points = normalisePoints(rawPoints).map((point, index) => ({
    ...point,
    phase: index * 0.73,
    drift: point.point_type === "synthetic" ? 3.5 : point.point_type === "correction" ? 8 : 5,
  }));
  state.clusters = payload.clusters || [];
  state.source = payload.source || "unknown";
  state.projection = payload.projection || "metadata_cluster";
  updateSelectOptions();
  updateLegend();
  fitToData();
}

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
});

window.addEventListener("pointerup", () => {
  state.dragging = false;
});

window.addEventListener("pointermove", (event) => {
  if (state.dragging) {
    const dx = event.clientX - state.lastX;
    const dy = event.clientY - state.lastY;
    state.targetRotationY += dx * 0.006;
    state.targetRotationX += dy * 0.004;
    state.targetRotationX = Math.max(-1.2, Math.min(1.0, state.targetRotationX));
    state.lastX = event.clientX;
    state.lastY = event.clientY;
  }
  state.hover = nearestPoint(event.clientX, event.clientY);
  document.body.style.cursor = state.hover ? "pointer" : state.dragging ? "grabbing" : "";
  updateTooltip(event);
});

canvas.addEventListener("click", (event) => {
  const point = nearestPoint(event.clientX, event.clientY);
  if (point) selectPoint(point);
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  state.targetZoom = Math.max(1.2, Math.min(8.5, state.targetZoom - event.deltaY * 0.004));
}, { passive: false });

controls.fit.addEventListener("click", fitToData);
controls.reset.addEventListener("click", resetCamera);
controls.fullscreen.addEventListener("click", () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
});

[controls.search, controls.relationship, controls.intent, controls.source, controls.synthetic, controls.neighbours, controls.spread].forEach((element) => {
  element.addEventListener("input", () => updateStats(state.projected.length));
});
controls.memoryType.addEventListener("input", () => {
  state.selected = null;
  state.neighbours = [];
  loadVectorMap().catch((error) => {
    inspector.innerHTML = `<p class="eyebrow">Vector Database</p><h2>Could not load map</h2><p class="warning">${escapeHtml(error.message)}</p>`;
  });
});
controls.colorMode.addEventListener("input", updateLegend);
window.addEventListener("resize", () => {
  resize();
  fitToData();
});

resize();
loadVectorMap().catch((error) => {
  inspector.innerHTML = `<p class="eyebrow">Vector Database</p><h2>Could not load map</h2><p class="warning">${escapeHtml(error.message)}</p>`;
});
requestAnimationFrame(render);
