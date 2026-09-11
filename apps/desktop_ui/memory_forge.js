(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
mountCyberPage({
  title: "Memory Forge",
  subtitle: "Training, correction, and vector rebuild pipeline with authority tier visibility and recent session state.",
  activePath: "/memory-forge",
  contentHtml: `<section class="cyber-grid two"><article class="cyber-panel"><h2>Forge State</h2><div id="forgeState" class="cyber-stack"></div></article><article class="cyber-panel"><h2>Recent Sessions</h2><div id="forgeRecent" class="cyber-stack"></div></article></section>`,
  onMounted: async () => {
    const payload = await cyberFetch("/api/memory-forge/status").catch(() => null);
    if (!payload) return;
    document.getElementById("forgeState").innerHTML = [
      `<div class="cyber-row"><strong>Corrections</strong><small>${escapeHtml(payload.corrections_count ?? "Unknown")}</small></div>`,
      `<div class="cyber-row"><strong>Training examples</strong><small>${escapeHtml(payload.training_examples_count ?? "Unknown")}</small></div>`,
      `<div class="cyber-row"><strong>Vector DB</strong><small>${escapeHtml(payload.vector_db_available ? "Available" : "Unknown")}</small></div>`,
      `<div class="cyber-row"><strong>Last rebuild</strong><small>${escapeHtml(payload.last_rebuild_status?.last_vector_rebuild || "Unknown")}</small></div>`,
    ].join("");
    document.getElementById("forgeRecent").innerHTML = (payload.recent_sessions || []).slice(0, 10).map((item) => `<div class="cyber-event"><strong>${escapeHtml(item.contact_name || item.id || "session")}</strong><small>${escapeHtml(item.timestamp || "Unknown")}</small><small>${escapeHtml(item.feedback_label || item.source || "Unknown")}</small></div>`).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
  },
});
})();
