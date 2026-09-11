(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
const renderValue = cyberOps.renderValue;
const statusClass = cyberOps.statusClass;

function card(label, value, tone = "") {
  return `<div class="cyber-pill" data-tone="${tone}"><span>${escapeHtml(label)}</span><strong>${renderValue(value)}</strong></div>`;
}

function row(label, value, tone = "") {
  return `<div class="cyber-row ${tone ? tone : ""}"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(value ?? "Unknown")}</small></div>`;
}

function eventRow(event) {
  const status = String(event.event_type || "");
  return `
    <div class="cyber-event ${statusClass(status)}">
      <strong>${escapeHtml(event.event_type || "unknown")}</strong>
      <small>${escapeHtml(event.timestamp || "Unknown")}</small>
      <small>${escapeHtml(event.execution_result || "Unknown")}</small>
    </div>
  `;
}

mountCyberPage({
  title: "Mission Control",
  subtitle: "Operational overview of the local assistant state, live health, current app, confidence, halt state, and recent event stream.",
  activePath: "/mission-control",
  contentHtml: `
    <section class="cyber-status-strip" id="statusStrip"></section>
    <section class="cyber-grid three">
      <article class="cyber-panel">
        <h2>Device Health</h2>
        <div id="leftColumn" class="cyber-stack"></div>
      </article>
      <article class="cyber-panel">
        <h2>System Twin</h2>
        <div id="centerColumn" class="cyber-stack"></div>
      </article>
      <article class="cyber-panel">
        <h2>Live Events</h2>
        <div id="eventStream" class="cyber-stack"></div>
      </article>
    </section>
    <section class="cyber-timeline">
      <h2>Bottom Pipeline Strip</h2>
      <div id="pipelineStrip" class="cyber-timeline-track"></div>
    </section>
  `,
  onMounted: async () => {
    const payload = await cyberFetch("/api/mission-control/status").catch(() => null);
    const strip = document.getElementById("statusStrip");
    const left = document.getElementById("leftColumn");
    const center = document.getElementById("centerColumn");
    const events = document.getElementById("eventStream");
    const pipeline = document.getElementById("pipelineStrip");

    if (!payload) {
      if (strip) strip.innerHTML = card("Status", "Metric unavailable", "warn");
      if (left) left.innerHTML = `<div class="cyber-empty">No trace recorded yet</div>`;
      if (center) center.innerHTML = `<div class="cyber-empty">No trace recorded yet</div>`;
      if (events) events.innerHTML = `<div class="cyber-empty">No trace recorded yet</div>`;
      if (pipeline) pipeline.innerHTML = `<div class="cyber-empty">No trace recorded yet</div>`;
      return;
    }

    const state = payload.state || {};
    const device = state.device || {};
    const automation = state.automation || {};
    const messages = payload.messages || {};
    const metrics = payload.metrics || {};
    const health = payload.health_cards || [];

    strip.innerHTML = [
      card("Connection", state.connection_status || "unknown", state.connected ? "good" : "warn"),
      card("Mode", automation.mode || "unknown", ""),
      card("Risk", state.risk_level || "unknown", state.risk_level === "halted" || state.risk_level === "high" ? "danger" : "warn"),
      card("Halt", state.halted ? "Yes" : "No", state.halted ? "danger" : "good"),
      card("Confidence", state.confidence == null ? "Unknown" : Number(state.confidence).toFixed(2), ""),
      card("Last action", state.last_action || "Unknown", ""),
    ].join("");

    left.innerHTML = [
      row("Connected device", device.app || "Unknown"),
      row("Current app", device.app || "Unknown"),
      row("Current screen", device.screen || "unknown"),
      row("Keyboard state", device.keyboard_visible === null ? "Unknown" : device.keyboard_visible ? "Visible" : "Hidden"),
      row("Automation mode", automation.mode || "unknown"),
      row("Emergency stop", state.emergency_stop ? "Engaged" : "Clear", state.emergency_stop ? "danger" : "good"),
      row("Confidence score", state.confidence == null ? "Unknown" : Number(state.confidence).toFixed(2)),
      row("Risk level", state.risk_level || "unknown"),
      row("Training rows", metrics.compose_validation ? metrics.compose_validation.total_runs : "Unknown"),
    ].join("");

    center.innerHTML = [
      row("Last observed incoming", messages.last_observed_incoming || "No trace recorded yet"),
      row("Last drafted reply", messages.last_drafted_reply || "No trace recorded yet"),
      row("Last sent reply", messages.last_sent_reply || "No trace recorded yet"),
      row("Approval state", state.final_decision || "unknown"),
      row("Blocked reason", state.blocked_reason || "Unknown"),
      row("Pipeline summary", payload.pipeline && payload.pipeline.length ? `${payload.pipeline.length} stages available` : "No trace recorded yet"),
      `<div class="cyber-row"><strong>Metrics snapshot</strong><pre class="cyber-monobox">${escapeHtml(JSON.stringify(metrics, null, 2))}</pre></div>`,
    ].join("");

    events.innerHTML = (payload.events || []).slice(0, 12).map(eventRow).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
    pipeline.innerHTML = (payload.pipeline || []).map((stage) => `
      <div class="cyber-step ${statusClass(stage.status)}">
        <strong>${escapeHtml(stage.label || stage.id)}</strong>
        <small>${escapeHtml(stage.status || "unknown")}</small>
        <small>${escapeHtml(stage.output_summary || "Unknown")}</small>
      </div>
    `).join("") || `<div class="cyber-empty">No trace recorded yet</div>`;
  },
});
})();
