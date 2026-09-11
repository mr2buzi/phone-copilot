const DEFAULT_ENDPOINT = "http://127.0.0.1:8765";
const TONE_PARAMETERS = {
  normal: { risk_tolerance: 0.0, directness: 0.7, warmth: 0.5, banter: 0.4, slang_level: 0.6 },
  bolder: { risk_tolerance: 0.25, directness: 0.82, warmth: 0.58, banter: 0.62, slang_level: 0.72 },
  riskier: { risk_tolerance: 0.4, directness: 0.9, warmth: 0.62, banter: 0.75, slang_level: 0.78 },
};
const LONG_REQUEST_PORT = "phone-copilot-long-request";
const MANUAL_DRAFT_TIMEOUT_MS = 300000;
const AUTO_SCAN_DRAFT_TIMEOUT_MS = 300000;

async function readSettings() {
  const stored = await chrome.storage.local.get({
    endpoint: DEFAULT_ENDPOINT,
    mode: "review",
    relationshipType: "unknown",
    toneRisk: "bolder",
    archiveAllowedContactNames: [],
  });
  const toneRisk = TONE_PARAMETERS[String(stored.toneRisk || "bolder")] ? String(stored.toneRisk || "bolder") : "bolder";
  return {
    endpoint: String(stored.endpoint || DEFAULT_ENDPOINT).replace(/\/+$/, ""),
    mode: String(stored.mode || "review"),
    relationshipType: String(stored.relationshipType || "unknown"),
    toneRisk,
    archiveAllowedContactNames: Array.isArray(stored.archiveAllowedContactNames) ? stored.archiveAllowedContactNames.map((name) => String(name || "")).filter(Boolean) : [],
  };
}

async function saveSettings(payload) {
  const update = {};
  if (payload.endpoint) update.endpoint = String(payload.endpoint).replace(/\/+$/, "");
  if (payload.mode) update.mode = String(payload.mode);
  if (payload.relationshipType) update.relationshipType = String(payload.relationshipType);
  if (payload.toneRisk) update.toneRisk = String(payload.toneRisk);
  if (Array.isArray(payload.archiveAllowedContactNames)) {
    update.archiveAllowedContactNames = payload.archiveAllowedContactNames.map((name) => String(name || "")).filter(Boolean);
  }
  await chrome.storage.local.set(update);
  return readSettings();
}

async function requestDraft(payload) {
  const settings = await readSettings();
  const source = String(payload.source || "");
  const timeoutMs = source.endsWith("-auto") ? AUTO_SCAN_DRAFT_TIMEOUT_MS : MANUAL_DRAFT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(`${settings.endpoint}/api/web/whatsapp/draft`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        request_id: payload.requestId || "",
        thread_id: payload.threadId || "",
        contact_name: payload.contactName || "",
        messages: payload.messages || [],
        relationship_type: settings.relationshipType,
        intent_type: "auto",
        mode: settings.mode,
        diversity_mode: "natural",
        avoid_candidates: payload.avoidCandidates || [],
        parameters: TONE_PARAMETERS[settings.toneRisk] || TONE_PARAMETERS.bolder,
      }),
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(`Phone Copilot controller timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || `Phone Copilot returned ${response.status}`);
  }
  return data;
}

async function saveMemory(payload) {
  const settings = await readSettings();
  const response = await fetch(`${settings.endpoint}/api/web/whatsapp/memory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contact_name: payload.contactName || "",
      relationship_type: settings.relationshipType,
      question: payload.question || "",
      answer: payload.answer || "",
      messages: payload.messages || [],
      source: "whatsapp_web_extension",
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || `Phone Copilot returned ${response.status}`);
  }
  return data;
}

function debuggerCall(target, method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand(target, method, params, (result) => {
      const error = chrome.runtime.lastError;
      if (error) {
        reject(new Error(error.message));
        return;
      }
      resolve(result);
    });
  });
}

function debuggerAttach(target) {
  return new Promise((resolve, reject) => {
    chrome.debugger.attach(target, "1.3", () => {
      const error = chrome.runtime.lastError;
      if (error) {
        reject(new Error(error.message));
        return;
      }
      resolve();
    });
  });
}

function debuggerDetach(target) {
  return new Promise((resolve) => {
    chrome.debugger.detach(target, () => resolve());
  });
}

async function trustedClick(payload, senderContext = {}) {
  const tabId = Number(senderContext.tabId || 0);
  const x = Number(payload?.x);
  const y = Number(payload?.y);
  if (!chrome.debugger?.attach || !chrome.debugger?.sendCommand || !chrome.debugger?.detach) {
    throw new Error("debugger_api_unavailable");
  }
  if (!tabId) {
    throw new Error("trusted click requires an active tab id");
  }
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error("trusted click requires finite x/y coordinates");
  }
  const target = { tabId };
  await debuggerAttach(target);
  try {
    await debuggerCall(target, "Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x,
      y,
    });
    await debuggerCall(target, "Input.dispatchMouseEvent", {
      type: "mousePressed",
      x,
      y,
      button: "left",
      clickCount: 1,
    });
    await debuggerCall(target, "Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x,
      y,
      button: "left",
      clickCount: 1,
    });
  } finally {
    await debuggerDetach(target);
  }
  return { clicked: true, x, y };
}

async function handleRuntimeMessage(message, senderContext = {}) {
  if (!message || typeof message.type !== "string") {
    return { ok: false, error: "unknown message type" };
  }
  if (message.type === "pc:get-settings") {
    return { ok: true, settings: await readSettings() };
  }
  if (message.type === "pc:save-settings") {
    return { ok: true, settings: await saveSettings(message.payload || {}) };
  }
  if (message.type === "pc:draft") {
    return { ok: true, data: await requestDraft(message.payload || {}) };
  }
  if (message.type === "pc:save-memory") {
    return { ok: true, data: await saveMemory(message.payload || {}) };
  }
  if (message.type === "pc:trusted-click") {
    return { ok: true, data: await trustedClick(message.payload || {}, senderContext) };
  }
  return { ok: false, error: "unknown message type" };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "pc:draft" || message?.type === "pc:save-memory") {
    sendResponse({
      ok: false,
      error: "Phone Copilot extension was updated. Refresh WhatsApp Web so archive scan can use the long-lived controller channel.",
    });
    return false;
  }
  handleRuntimeMessage(message, { tabId: sender?.tab?.id || 0 })
    .then((response) => sendResponse(response))
    .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
  return true;
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== LONG_REQUEST_PORT) return;
  let disconnected = false;
  const senderContext = { tabId: port.sender?.tab?.id || 0 };
  port.onDisconnect.addListener(() => {
    disconnected = true;
  });
  port.onMessage.addListener((message) => {
    if (message?.type === "pc:ping") {
      if (!disconnected) port.postMessage({ id: message.id || "", ok: true, pong: true });
      return;
    }
    const id = message && message.id ? String(message.id) : "";
    handleRuntimeMessage(message, senderContext)
      .then((response) => {
        if (!disconnected) port.postMessage({ id, ...response });
      })
      .catch((error) => {
        if (!disconnected) port.postMessage({ id, ok: false, error: error.message || String(error) });
      });
  });
});
