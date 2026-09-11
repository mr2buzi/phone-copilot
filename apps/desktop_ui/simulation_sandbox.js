(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
mountCyberPage({
  title: "Simulation Sandbox",
  subtitle: "Offline drafting and policy simulation. This page never touches the phone or calls ADB.",
  activePath: "/simulation-sandbox",
  contentHtml: `
    <section class="cyber-grid two">
      <article class="cyber-panel">
        <h2>Scenario</h2>
        <textarea id="simIncoming" class="text-input correction-textarea">hii</textarea>
        <textarea id="simContext" class="text-input correction-textarea" placeholder="Context lines"></textarea>
        <div class="cyber-button-row"><button class="cyber-button primary" id="runSim">Run Simulation</button></div>
      </article>
      <article class="cyber-panel"><h2>Result</h2><div id="simResult" class="cyber-stack"></div></article>
    </section>
  `,
  onMounted: () => {
    document.getElementById("runSim").onclick = async () => {
      const incoming = document.getElementById("simIncoming").value;
      const context = document.getElementById("simContext").value.split("\n").map((line) => line.trim()).filter(Boolean);
      const payload = await cyberFetch("/api/simulation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incoming, context, relationship_type: "unknown", intent_type: "auto", automation_mode: "review" }),
      }).catch(() => null);
      const result = document.getElementById("simResult");
      if (!payload) {
        result.innerHTML = `<div class="cyber-empty">Metric unavailable</div>`;
        return;
      }
      result.innerHTML = [
        `<div class="cyber-row"><strong>Decision</strong><small>${escapeHtml(payload.policy_result?.decision || "Unknown")}</small></div>`,
        `<div class="cyber-row"><strong>Offline</strong><small>${payload.offline ? "Yes" : "No"}</small></div>`,
        `<div class="cyber-row"><strong>Selected candidate</strong><small>${escapeHtml(payload.selected_candidate?.text || "Unknown")}</small></div>`,
        `<div class="cyber-row"><strong>Selected memories</strong><small>${escapeHtml((payload.selected_memories || []).length || "0")}</small></div>`,
      ].join("");
    };
  },
});
})();
