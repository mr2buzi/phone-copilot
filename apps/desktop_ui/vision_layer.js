(() => {
const cyberOps = window.CyberOps;
const mountCyberPage = cyberOps.mountCyberPage;
const cyberFetch = cyberOps.cyberFetch;
const escapeHtml = cyberOps.escapeHtml;
const statusClass = cyberOps.statusClass;

mountCyberPage({
  title: "Vision Layer",
  subtitle: "Latest screenshot with classifier metadata, OCR text, keyboard status, and bounded overlay targets.",
  activePath: "/vision-layer",
  contentHtml: `
    <section class="cyber-grid two">
      <article class="cyber-panel">
        <h2>Phone Frame</h2>
        <div class="cyber-button-row" style="margin-bottom:12px;">
          <span class="cyber-badge good">OCR</span>
          <span class="cyber-badge good">Keyboard</span>
          <span class="cyber-badge good">Action Targets</span>
          <span class="cyber-badge warn">Classifier Debug</span>
        </div>
        <div class="cyber-phone" id="phoneFrame">
          <img id="visionScreenshot" alt="Latest screenshot" />
        </div>
      </article>
      <article class="cyber-inspector">
        <h3>Classifier Metadata</h3>
        <div id="visionMeta" class="cyber-stack"></div>
      </article>
    </section>
    <section class="cyber-panel">
      <h2>OCR / Parse Output</h2>
      <pre id="visionOcr" class="cyber-monobox"></pre>
    </section>
  `,
  onMounted: async () => {
    const payload = await cyberFetch("/api/vision-layer/latest").catch(() => null);
    const shot = document.getElementById("visionScreenshot");
    const meta = document.getElementById("visionMeta");
    const ocr = document.getElementById("visionOcr");
    if (!payload) {
      meta.innerHTML = `<div class="cyber-empty">No trace recorded yet</div>`;
      ocr.textContent = "No trace recorded yet";
      return;
    }
    if (payload.screenshot_url) shot.src = payload.screenshot_url;
    meta.innerHTML = [
      `<div class="cyber-row"><strong>Screen</strong><small>${escapeHtml(payload.classification?.screen || "unknown")}</small></div>`,
      `<div class="cyber-row"><strong>Confidence</strong><small>${escapeHtml(payload.classification_confidence == null ? "Unknown" : Number(payload.classification_confidence).toFixed(2))}</small></div>`,
      `<div class="cyber-row"><strong>Keyboard</strong><small>${escapeHtml(payload.keyboard_region ? "Visible" : "Unknown")}</small></div>`,
      `<div class="cyber-row"><strong>Low confidence</strong><small>${escapeHtml(payload.low_confidence_warning ? "Yes" : "No")}</small></div>`,
      `<div class="cyber-row"><strong>Fixture debug</strong><small>${escapeHtml(payload.fixture_debug?.status || "unknown")}</small></div>`,
    ].join("");
    ocr.textContent = JSON.stringify({
      ocr_text: payload.ocr_text || [],
      action_targets: payload.action_targets || [],
      keyboard_region: payload.keyboard_region || null,
      debug_info: payload.fixture_debug?.debug_info || {},
    }, null, 2);
    const frame = document.getElementById("phoneFrame");
    (payload.action_targets || []).forEach((target) => {
      const box = document.createElement("div");
      box.className = `cyber-overlay-box ${statusClass(target.type)}`;
      box.style.left = `${Math.max(0, target.left || 0)}px`;
      box.style.top = `${Math.max(0, target.top || 0)}px`;
      box.style.width = `${Math.max(12, (target.right || 0) - (target.left || 0))}px`;
      box.style.height = `${Math.max(12, (target.bottom || 0) - (target.top || 0))}px`;
      box.innerHTML = `<span>${escapeHtml(target.label || target.type || "target")}</span>`;
      frame.appendChild(box);
    });
  },
});
})();
