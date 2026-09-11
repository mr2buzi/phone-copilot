const CYBER_PAGES = [
  { href: "/mission-control", label: "Mission Control" },
  { href: "/execution-trace", label: "Execution Trace" },
  { href: "/signal-replay", label: "Signal Replay" },
  { href: "/vision-layer", label: "Vision Layer" },
  { href: "/memory-inspector", label: "Memory Inspector" },
  { href: "/contact-graph", label: "Contact Graph" },
  { href: "/failure-atlas", label: "Failure Atlas" },
  { href: "/risk-engine", label: "Risk Engine" },
  { href: "/memory-forge", label: "Memory Forge" },
  { href: "/simulation-sandbox", label: "Simulation Sandbox" },
  { href: "/ai-core", label: "AI Core" },
  { href: "/ai-core/vector-map", label: "Vector Map" },
  { href: "/training", label: "Training" },
];

function pathMatches(pathname, href) {
  return pathname === href || (href !== "/" && pathname.startsWith(href));
}

function cyberShell(pageTitle, subtitle, activePath, content) {
  const contentHtml = typeof content === "string" ? content : content instanceof HTMLElement ? content.outerHTML : "";
  const shell = document.createElement("div");
  shell.className = "cyber-shell";
  shell.innerHTML = `
    <div class="cyber-scanline"></div>
    <div class="cyber-shell-frame">
      <aside class="cyber-sidebar">
        <div class="cyber-brand">
          <strong>phone-copilot</strong>
          <span>Cyber AI operations console</span>
        </div>
        <div class="cyber-nav-group">
          <div class="cyber-nav-label">Cyber Ops</div>
          ${CYBER_PAGES.map((item) => `
            <a class="cyber-nav-link ${pathMatches(activePath, item.href) ? "active" : ""}" href="${item.href}">
              <span class="cyber-nav-dot"></span>
              <span>${item.label}</span>
            </a>
          `).join("")}
        </div>
      </aside>
      <main class="cyber-main">
        <section class="cyber-topbar">
          <div class="cyber-title-block">
            <div class="cyber-kicker">Supervised local-first system</div>
            <h1 class="cyber-title">${pageTitle}</h1>
            <p class="cyber-subtitle">${subtitle}</p>
          </div>
          <div class="cyber-button-row">
            <a class="cyber-button secondary" href="/">Live Phone</a>
            <a class="cyber-button secondary" href="/training/simple">Simple Training</a>
          </div>
        </section>
        ${contentHtml}
      </main>
    </div>
  `;
  return shell;
}

async function cyberFetch(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || payload.error || "Request failed.");
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderValue(value) {
  if (value === null || value === undefined || value === "") {
    return "Unknown";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(2).replace(/\.00$/, "") : "Unknown";
  }
  return escapeHtml(value);
}

function setAppShell({ title, subtitle, activePath, contentSelector }) {
  const mount = document.getElementById("app");
  const content = document.querySelector(contentSelector);
  if (!mount || !content) {
    return;
  }
  mount.replaceWith(cyberShell(title, subtitle, activePath, content.outerHTML));
}

function mountCyberPage({ title, subtitle, activePath, contentHtml, onMounted }) {
  const mount = document.getElementById("app");
  if (!mount) {
    return;
  }
  const shell = cyberShell(title, subtitle, activePath, contentHtml);
  mount.replaceWith(shell);
  if (typeof onMounted === "function") {
    onMounted(document);
  }
}

function statusClass(status) {
  if (["failed", "halted", "blocked"].includes(status)) return "danger";
  if (["warning", "warn", "review"].includes(status)) return "warn";
  if (["passed", "good", "safe", "observed", "available"].includes(status)) return "good";
  return "";
}

window.CyberOps = {
  cyberShell,
  cyberFetch,
  escapeHtml,
  renderValue,
  setAppShell,
  mountCyberPage,
  statusClass,
};
