async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || "Request failed");
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function showBanner(message) {
  document.getElementById("haltBanner").classList.remove("hidden");
  document.getElementById("haltBannerText").textContent = message;
}

function keyValueList(data) {
  return Object.entries(data || {}).map(([key, value]) => `
    <div class="debug-example">
      <small>${escapeHtml(key)}</small>
      <p>${escapeHtml(typeof value === "object" ? JSON.stringify(value) : value)}</p>
    </div>
  `).join("");
}

function renderProviders(status) {
  const routing = status.provider_routing || {};
  const providers = status.providers || {};
  const rows = Object.entries(providers).map(([name, data]) => `
    <div class="debug-example">
      <small>${escapeHtml(name)} | configured=${escapeHtml(Boolean(data.configured))}</small>
      <p>${escapeHtml(data.model || data.base_url || "")}</p>
    </div>
  `).join("");
  return `
    ${keyValueList({
      draft_provider: routing.draft_provider,
      fast_provider: routing.fast_provider,
      router_provider: routing.router_provider,
      private_provider: routing.private_provider,
      external_api_enabled: routing.external_api_enabled,
      external_api_allow_sensitive: routing.external_api_allow_sensitive,
    })}
    ${rows}
  `;
}

async function loadCore() {
  try {
    const status = await api("/api/ai-core/status");
    document.getElementById("coreStats").innerHTML = `
      <div><span>Active rows</span><strong>${Number(status.training_data?.active_rows || 0)}</strong></div>
      <div><span>Corrections</span><strong>${Number(status.training_data?.corrections_count || 0)}</strong></div>
      <div><span>Synthetic</span><strong>${Number(status.training_data?.synthetic_examples_count || 0)}</strong></div>
      <div><span>Vector rows</span><strong>${Number(status.vector_database?.indexed_count || 0)}</strong></div>
    `;
    document.getElementById("coreFiles").innerHTML = keyValueList(status.files);
    document.getElementById("trainingData").innerHTML = `
      ${keyValueList({
        active_rows: status.training_data?.active_rows,
        real_examples: status.training_data?.real_examples_count,
        synthetic_examples: status.training_data?.synthetic_examples_count,
        corrections: status.training_data?.corrections_count,
      })}
      ${(status.training_data?.files || []).map((file) => `
        <div class="debug-example">
          <small>${escapeHtml(file.name)} | ${Number(file.rows || 0)} rows | ${Number(file.bytes || 0)} bytes</small>
          <p>${escapeHtml(file.path)}</p>
        </div>
      `).join("")}
    `;
    document.getElementById("vectorData").innerHTML = keyValueList(status.vector_database);
    document.getElementById("identityData").innerHTML = keyValueList(status.identity || {});
    document.getElementById("retrievalConfig").textContent = JSON.stringify(status.retrieval || {}, null, 2);
    document.getElementById("safetyData").innerHTML = keyValueList(status.safety);
    document.getElementById("providerData").innerHTML = renderProviders(status);
  } catch (error) {
    showBanner(error.message);
  }
}

async function reloadIdentityPack() {
  try {
    const payload = await api("/api/ai-core/reload-identity-pack", {method: "POST"});
    document.getElementById("identityData").innerHTML = keyValueList(payload);
  } catch (error) {
    showBanner(error.message);
  }
}

async function runProviderCompare() {
  const incoming = document.getElementById("providerCompareIncoming").value.trim();
  const context = document.getElementById("providerCompareContext").value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const providers = document.getElementById("providerCompareProviders").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const target = document.getElementById("providerCompareResults");
  target.innerHTML = `<div class="debug-example"><p>Running comparison...</p></div>`;
  try {
    const payload = await api("/api/ai-core/provider-compare", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        incoming,
        context,
        relationship_type: "close_friend",
        intent_type: "planning",
        providers,
      }),
    });
    target.innerHTML = (payload.results || []).map((result) => `
      <div class="debug-example">
        <small>${escapeHtml(result.provider)} | ${escapeHtml(result.model)} | latency=${escapeHtml(result.latency_ms ?? "")}ms</small>
        <p>${escapeHtml(result.error || result.reply || "No reply")}</p>
        <small>style=${escapeHtml(result.style_score)} assistant=${escapeHtml(result.assistant_likeness_score)} risk=${escapeHtml(result.risk_score)} external=${escapeHtml(result.external_api_used)} blocked=${escapeHtml(result.external_api_blocked)}</small>
      </div>
    `).join("");
  } catch (error) {
    target.innerHTML = `<div class="debug-example"><p>${escapeHtml(error.message)}</p></div>`;
  }
}

document.getElementById("refreshCoreButton").onclick = loadCore;
document.getElementById("reloadIdentityButton").onclick = reloadIdentityPack;
document.getElementById("providerCompareButton").onclick = runProviderCompare;
loadCore();
