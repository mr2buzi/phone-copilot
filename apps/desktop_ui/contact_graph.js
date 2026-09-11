(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
mountCyberPage({
  title: "Contact Graph",
  subtitle: "Constellation of contact profiles, relationship type, interaction density, and blacklist state.",
  activePath: "/contact-graph",
  contentHtml: `<section class="cyber-grid two"><article class="cyber-panel"><h2>Contact Nodes</h2><div id="graphNodes" class="cyber-stack"></div></article><article class="cyber-inspector"><h3>Inspector</h3><div id="graphDetail" class="cyber-stack"></div></article></section>`,
  onMounted: async () => {
    const payload = await cyberFetch("/api/contact-graph/data").catch(() => ({ nodes: [] }));
    const nodes = document.getElementById("graphNodes");
    const detail = document.getElementById("graphDetail");
    nodes.innerHTML = (payload.nodes || []).map((node) => `<button class="cyber-row" type="button" data-id="${escapeHtml(node.id)}" style="text-align:left;"><strong>${escapeHtml(node.contact_name || "Unknown")}</strong><small>${escapeHtml(node.relationship_type || "unknown")} · ${escapeHtml(node.interaction_count ?? "0")} interactions</small></button>`).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
    nodes.querySelectorAll("[data-id]").forEach((button) => button.addEventListener("click", () => {
      const node = (payload.nodes || []).find((item) => item.id === button.dataset.id);
      if (!node) return;
      detail.innerHTML = `<div class="cyber-row"><strong>${escapeHtml(node.contact_name || "Unknown")}</strong><small>${escapeHtml(node.relationship_type || "unknown")}</small></div><div class="cyber-row"><strong>Blacklist</strong><small>${node.blacklisted ? "Restricted" : "Clear"}</small></div><div class="cyber-row"><strong>Density</strong><small>${escapeHtml(node.interaction_count ?? "Unknown")}</small></div>`;
    }));
  },
});
})();
