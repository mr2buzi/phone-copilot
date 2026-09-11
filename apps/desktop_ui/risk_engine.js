(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
mountCyberPage({
  title: "Risk Engine",
  subtitle: "Current safety dimensions, decision state, and confidence waveform over recent iterations.",
  activePath: "/risk-engine",
  contentHtml: `<section class="cyber-grid two"><article class="cyber-panel"><h2>Decision</h2><div id="riskDecision" class="cyber-stack"></div><h2>Dimensions</h2><div id="riskDims" class="cyber-stack"></div></article><article class="cyber-panel"><h2>Confidence Waveform</h2><div id="riskWave" class="cyber-waveform"></div></article></section>`,
  onMounted: async () => {
    const payload = await cyberFetch("/api/risk-engine/status").catch(() => null);
    if (!payload) return;
    document.getElementById("riskDecision").innerHTML = `<div class="cyber-row"><strong>${escapeHtml(payload.decision || "UNKNOWN")}</strong><small>${escapeHtml(payload.risk_level || "unknown")}</small></div>`;
    document.getElementById("riskDims").innerHTML = Object.entries(payload.dimensions || {}).map(([key, value]) => `<div class="cyber-row"><strong>${escapeHtml(key)}</strong><small>${escapeHtml(value == null ? "Unknown" : Number(value).toFixed ? Number(value).toFixed(2) : value)}</small></div>`).join("");
    document.getElementById("riskWave").innerHTML = (payload.waveform || []).slice(-12).map((point) => `<div><small>${escapeHtml(point.timestamp || "")}</small><div class="cyber-wavebar"><span style="width:${Math.max(0, Math.min(100, (point.screen_confidence || 0) * 100))}%;"></span></div></div>`).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
  },
});
})();
