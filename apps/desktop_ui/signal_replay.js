(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
const statusClass = cyberOps.statusClass;
let selectedRunId = null;

mountCyberPage({
  title: "Signal Replay",
  subtitle: "Forensic mission playback built from real log entries, screenshots, and action metadata.",
  activePath: "/signal-replay",
  contentHtml: `
    <section class="cyber-grid two">
      <article class="cyber-panel">
        <h2>Replay View</h2>
        <div class="cyber-split">
          <div class="cyber-phone"><img id="replayScreenshot" alt="Replay screenshot" /></div>
          <div class="cyber-inspector">
            <h3>Replay Inspector</h3>
            <div id="replayInspector" class="cyber-stack"></div>
          </div>
        </div>
      </article>
      <article class="cyber-panel">
        <h2>Event List</h2>
        <div id="replayEvents" class="cyber-stack"></div>
      </article>
    </section>
    <section class="cyber-timeline">
      <div class="cyber-button-row">
        <button class="cyber-button secondary" id="prevRun">Previous Event</button>
        <button class="cyber-button secondary" id="nextRun">Next Event</button>
        <button class="cyber-button primary" id="playRun">Play</button>
        <button class="cyber-button secondary" id="pauseRun">Pause</button>
      </div>
      <div id="runTimeline" class="cyber-timeline-track"></div>
    </section>
  `,
  onMounted: async () => {
    const runs = await cyberFetch("/api/signal-replay/runs?limit=25").catch(() => ({ runs: [] }));
    const list = runs.runs || [];
    const events = document.getElementById("replayEvents");
    const inspector = document.getElementById("replayInspector");
    const screenshot = document.getElementById("replayScreenshot");
    const timeline = document.getElementById("runTimeline");
    const renderRun = async (runId) => {
      if (!runId) return;
      const payload = await cyberFetch(`/api/signal-replay/runs/${encodeURIComponent(runId)}`).catch(() => null);
      if (!payload) return;
      selectedRunId = runId;
      screenshot.src = payload.before_screenshot_url || payload.after_screenshot_url || "";
      inspector.innerHTML = `
        <div class="cyber-row"><strong>Timestamp</strong><small>${escapeHtml(payload.timestamp || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Planner</strong><small>${escapeHtml(payload.planner_decision || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Approval</strong><small>${escapeHtml(payload.approval_decision || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Outcome</strong><small>${escapeHtml(payload.execution_result || "Unknown")}</small></div>
        <div class="cyber-row"><strong>Failure point</strong><small>${escapeHtml(payload.failure_point || "Unknown")}</small></div>
      `;
      events.innerHTML = (payload.events || []).map((entry) => `
        <div class="cyber-event ${statusClass(entry.label || entry.event_type)}">
          <strong>${escapeHtml(entry.label || entry.event_type || "event")}</strong>
          <small>${escapeHtml(entry.timestamp || "Unknown")}</small>
          <small>${escapeHtml(entry.summary || entry.reason || "Unknown")}</small>
        </div>
      `).join("") || `<div class="cyber-empty">No replay data available</div>`;
      timeline.innerHTML = (payload.events || []).map((entry) => `
        <div class="cyber-step ${statusClass(entry.label || entry.event_type)}">
          <strong>${escapeHtml(entry.label || entry.event_type || "event")}</strong>
          <small>${escapeHtml(entry.timestamp || "Unknown")}</small>
        </div>
      `).join("") || `<div class="cyber-empty">No replay data available</div>`;
    };
    const firstRun = list[0];
    if (firstRun) renderRun(firstRun.run_id);
    events.innerHTML = list.map((run) => `
      <button type="button" class="cyber-step" data-run="${escapeHtml(run.run_id)}">
        <strong>${escapeHtml(run.event_type || "event")}</strong>
        <small>${escapeHtml(run.timestamp || "Unknown")}</small>
        <small>${escapeHtml(run.outcome || "Unknown")}</small>
      </button>
    `).join("") || `<div class="cyber-empty">No replay data available</div>`;
    events.querySelectorAll("[data-run]").forEach((button) => button.addEventListener("click", () => renderRun(button.dataset.run)));
    document.getElementById("prevRun").onclick = () => {
      const index = list.findIndex((item) => item.run_id === selectedRunId);
      if (index > 0) renderRun(list[index - 1].run_id);
    };
    document.getElementById("nextRun").onclick = () => {
      const index = list.findIndex((item) => item.run_id === selectedRunId);
      if (index >= 0 && index < list.length - 1) renderRun(list[index + 1].run_id);
    };
    document.getElementById("playRun").onclick = () => {};
    document.getElementById("pauseRun").onclick = () => {};
  },
});
})();
