(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
const statusClass = cyberOps.statusClass;

mountCyberPage({
  title: "Execution Trace",
  subtitle: "Stage-by-stage view of the active observation, retrieval, generation, policy, approval, execution, and verification path.",
  activePath: "/execution-trace",
  contentHtml: `
    <section class="cyber-grid two">
      <article class="cyber-panel">
        <h2>Glowing Pipeline</h2>
        <div id="stageRail" class="cyber-stack"></div>
      </article>
      <article class="cyber-inspector">
        <h3>Detail Inspector</h3>
        <div id="traceInspector" class="cyber-stack"></div>
      </article>
    </section>
    <section class="cyber-panel">
      <h2>Execution Trace Tree</h2>
      <div id="traceTree" class="cyber-stack"></div>
    </section>
  `,
  onMounted: async () => {
    const payload = await cyberFetch("/api/execution-trace/latest").catch(() => null);
    const rail = document.getElementById("stageRail");
    const inspector = document.getElementById("traceInspector");
    const tree = document.getElementById("traceTree");
    const stages = payload?.stages || [];
    const selectStage = (stage) => {
      inspector.innerHTML = `
        <div class="cyber-row"><strong>${escapeHtml(stage.label || stage.id || "Unknown")}</strong><small>${escapeHtml(stage.status || "unknown")}</small></div>
        <div class="cyber-row"><strong>Duration</strong><small>${escapeHtml(stage.duration_ms == null ? "Unknown" : `${Number(stage.duration_ms).toFixed(2)} ms`)}</small></div>
        <div class="cyber-row"><strong>Confidence</strong><small>${escapeHtml(stage.confidence == null ? "Unknown" : Number(stage.confidence).toFixed(2))}</small></div>
        <div class="cyber-row"><strong>Output</strong><small>${escapeHtml(stage.output_summary || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Failure</strong><small>${escapeHtml(stage.failure_reason || "Unknown")}</small></div>
      `;
    };
    rail.innerHTML = stages.map((stage, index) => `
      <button type="button" class="cyber-step ${statusClass(stage.status)} ${index === 0 ? "selected" : ""}" data-index="${index}">
        <strong>${escapeHtml(stage.label || stage.id)}</strong>
        <small>${escapeHtml(stage.status || "unknown")}</small>
        <small>${escapeHtml(stage.output_summary || "Unknown")}</small>
      </button>
    `).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
    rail.querySelectorAll("button[data-index]").forEach((button) => {
      button.addEventListener("click", () => {
        rail.querySelectorAll(".selected").forEach((node) => node.classList.remove("selected"));
        button.classList.add("selected");
        selectStage(stages[Number(button.dataset.index)]);
      });
    });
    if (stages[0]) selectStage(stages[0]);
    tree.innerHTML = (payload?.tree || []).map((node) => `
      <div class="cyber-row">
        <strong>${escapeHtml(node.id || "unknown")}</strong>
        <small>${escapeHtml(node.status || "unknown")}</small>
        <small>${escapeHtml((node.children || []).map((child) => child.label).join(" / ") || "Unknown")}</small>
      </div>
    `).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
  },
});
})();
