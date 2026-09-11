(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;

mountCyberPage({
  title: "Memory Inspector",
  subtitle: "Filtered memory list, selected detail pane, nearest neighbours, and a lexical versus vector retrieval battle view.",
  activePath: "/memory-inspector",
  contentHtml: `
    <section class="cyber-grid three">
      <article class="cyber-panel"><h2>Memory Index</h2><div id="memoryList" class="cyber-stack"></div></article>
      <article class="cyber-inspector"><h3>Selected Memory</h3><div id="memoryDetail" class="cyber-stack"></div></article>
      <article class="cyber-panel"><h2>Retrieval Battle</h2><div id="retrievalBattle" class="cyber-stack"></div></article>
    </section>
  `,
  onMounted: async () => {
    const data = await cyberFetch("/api/memory-inspector/data?limit=50").catch(() => ({ examples: [] }));
    const list = document.getElementById("memoryList");
    const detail = document.getElementById("memoryDetail");
    const battle = document.getElementById("retrievalBattle");
    const examples = data.examples || [];
    const pick = async (id) => {
      const payload = await cyberFetch(`/api/memory-inspector/example/${encodeURIComponent(id)}`).catch(() => null);
      if (!payload) return;
      detail.innerHTML = `
        <div class="cyber-row"><strong>${escapeHtml(payload.example.contact_name || "Unknown")}</strong><small>${escapeHtml(payload.example.intent_type || "unknown")} / ${escapeHtml(payload.example.relationship_type || "unknown")}</small></div>
        <div class="cyber-row"><strong>Authority</strong><small>${escapeHtml(payload.example.authority || "unknown")}</small></div>
        <div class="cyber-row"><strong>Incoming</strong><small>${escapeHtml(payload.example.incoming || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Reply</strong><small>${escapeHtml(payload.example.my_reply || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Neighbours</strong><small>${escapeHtml((payload.nearest_neighbours || []).length ? `${payload.nearest_neighbours.length} nearby rows` : "No trace recorded yet")}</small></div>
      `;
    };
    list.innerHTML = examples.map((example) => `
      <button type="button" class="cyber-row" data-id="${escapeHtml(example.id)}" style="text-align:left;">
        <strong>${escapeHtml(example.contact_name || "Unknown")}</strong>
        <small>${escapeHtml(example.intent_type || "unknown")} · ${escapeHtml(example.authority || "unknown")}</small>
      </button>
    `).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
    list.querySelectorAll("[data-id]").forEach((button) => button.addEventListener("click", () => pick(button.dataset.id)));
    if (examples[0]) pick(examples[0].id);
    battle.innerHTML = `<div class="cyber-empty">Retrieval Battle View requires a test query.</div>`;
  },
});
})();
