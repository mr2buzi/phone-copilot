(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
mountCyberPage({
  title: "Failure Atlas",
  subtitle: "Reliability wall showing halt categories, pipeline heat, recovery traces, and failure timelines.",
  activePath: "/failure-atlas",
  contentHtml: `<section class="cyber-grid two"><article class="cyber-panel"><h2>Failure Categories</h2><div id="failureList" class="cyber-stack"></div></article><article class="cyber-panel"><h2>Timeline</h2><div id="failureTimeline" class="cyber-stack"></div></article></section>`,
  onMounted: async () => {
    const payload = await cyberFetch("/api/failure-atlas/stats").catch(() => ({ common_failure_categories: {}, timeline: [] }));
    document.getElementById("failureList").innerHTML = Object.entries(payload.common_failure_categories || {}).map(([key, value]) => `<div class="cyber-row"><strong>${escapeHtml(key)}</strong><small>${escapeHtml(value)} events</small></div>`).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
    document.getElementById("failureTimeline").innerHTML = (payload.timeline || []).map((entry) => `<div class="cyber-event warn"><strong>${escapeHtml(entry.event_type || "event")}</strong><small>${escapeHtml(entry.timestamp || "Unknown")}</small><small>${escapeHtml(entry.reason || "Unknown")}</small></div>`).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
  },
});
})();
