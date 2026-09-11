const PANEL_ID = "phone-copilot-wa-panel";
const STATUS_IDLE = "ready";
const THREAD_CONTEXT_TARGET = 220;
const ARCHIVE_SCAN_INTERVAL_MS = 25000;
const CONTENT_VERSION = "wa-trace-bridge-20260605-archive-only-allow-all-004012";
const LONG_REQUEST_PORT = "phone-copilot-long-request";
const LONG_REQUEST_TIMEOUT_MS = 300000;
const LONG_REQUEST_KEEPALIVE_MS = 15000;
const SCAN_DRAFT_TIMEOUT_MS = 100000;
const STALLED_SCAN_BACKOFF_MS = 10 * 60 * 1000;
let lastDraft = "";
let lastDraftSequence = [];
let lastDraftTrace = null;
let autoTimer = null;
let archiveScanTimer = null;
let archiveScanRunning = false;
let observer = null;
let lastAutoMessageKey = "";
let cachedThreadMessages = [];
let cachedContactName = "";
let runtimeUnavailable = false;
let messageOrderCounter = 0;
let activeDraftRequestId = "";
let draftInFlight = false;
let archiveAllowedContactNames = [];
const archiveProcessedKeys = new Set();
const stalledScanKeys = new Map();

function pcLog(message, payload = undefined) {
  if (payload === undefined) {
    console.log(`[PhoneCopilot] ${message}`);
    return;
  }
  console.log(`[PhoneCopilot] ${message}`, payload);
}

function installPageDebugBridge() {
  try {
    window.__phoneCopilotContentVersion = CONTENT_VERSION;
    if (document.documentElement) {
      document.documentElement.dataset.phoneCopilotContentVersion = CONTENT_VERSION;
    }
    const bridgeVersion = document.documentElement?.dataset?.phoneCopilotDebugBridgeVersion || "";
    if (bridgeVersion === CONTENT_VERSION) {
      pcLog("content loaded", { version: CONTENT_VERSION, bridge: "main_world" });
      return;
    }
    pcLog("content loaded", {
      version: CONTENT_VERSION,
      bridge: "manifest_main_world_missing",
      previous_bridge: bridgeVersion || "none",
      action: "reload extension and WhatsApp Web",
    });
  } catch (error) {
    pcLog("debug bridge install failed", error.message || String(error));
  }
}

function frontendFailureTrace(overrides = {}) {
  return {
    content_version: CONTENT_VERSION,
    created_at: new Date().toISOString(),
    request_id: "",
    thread_id: "",
    chat_name: "",
    stage: "frontend_failure",
    error: "",
    controller_called: false,
    response_received: false,
    ui_rendered: false,
    active_request_id: activeDraftRequestId,
    draft_in_flight: draftInFlight,
    latest_burst_preview: "",
    ...overrides,
  };
}

function updateTracePreview(trace) {
  const panel = document.getElementById(PANEL_ID);
  const versionNode = panel?.querySelector("[data-pc='debug-version']");
  const traceNode = panel?.querySelector("[data-pc='trace-preview']");
  if (versionNode) versionNode.textContent = `content ${CONTENT_VERSION}`;
  if (traceNode) {
    const request = trace?.request_id || "-";
    const stage = trace?.stage || trace?.final_decision || trace?.response_accepted_or_rejected || "-";
    const units = Array.isArray(trace?.final_reply_units) ? trace.final_reply_units.length : Number(trace?.rendered_draft_units_count || 0);
    traceNode.textContent = `trace ${request} | ${stage} | units ${units}`;
  }
}

async function storeDraftTrace(trace) {
  const stored = {
    content_version: CONTENT_VERSION,
    stored_at: new Date().toISOString(),
    ...(trace || {}),
  };
  lastDraftTrace = stored;
  window.__phoneCopilotLastDraftTrace = stored;
  updateTracePreview(stored);
  try {
    window.postMessage({
      source: "phone-copilot-content",
      type: "draft-trace",
      version: CONTENT_VERSION,
      trace: stored,
    }, "*");
  } catch (error) {
    pcLog("page trace bridge post failed", error.message || String(error));
  }
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
      await chrome.storage.local.set({
        phoneCopilotContentVersion: CONTENT_VERSION,
        phoneCopilotLastDraftTrace: stored,
      });
    }
  } catch (error) {
    pcLog("storage trace write failed", error.message || String(error));
  }
  pcLog("trace stored", {
    request_id: stored.request_id || "",
    thread_id: stored.thread_id || "",
    stage: stored.stage || stored.final_decision || stored.response_accepted_or_rejected || "",
  });
  return stored;
}

async function readStoredDraftTrace() {
  if (lastDraftTrace) return lastDraftTrace;
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
      const stored = await chrome.storage.local.get(["phoneCopilotLastDraftTrace"]);
      if (stored.phoneCopilotLastDraftTrace) {
        lastDraftTrace = stored.phoneCopilotLastDraftTrace;
        updateTracePreview(lastDraftTrace);
        return lastDraftTrace;
      }
    }
  } catch (error) {
    pcLog("storage trace read failed", error.message || String(error));
  }
  return null;
}

async function storeSuppressedDraftTrace(source) {
  if (lastDraftTrace?.request_id) {
    const count = Number(lastDraftTrace.suppressed_reentrant_attempts || 0) + 1;
    await storeDraftTrace({
      ...lastDraftTrace,
      suppressed_reentrant_attempts: count,
      last_suppressed_reentrant_source: source,
      last_suppressed_reentrant_at: new Date().toISOString(),
      active_request_id: activeDraftRequestId,
      draft_in_flight: draftInFlight,
    });
    return;
  }
  await storeDraftTrace(frontendFailureTrace({
    stage: "frontend_failure",
    error: "draft already running",
    response_accepted_or_rejected: "rejected",
    rejection_reason: "draft_already_running",
    controller_called: false,
    response_received: false,
    ui_rendered: false,
  }));
}

async function copyLastDraftTrace() {
  const trace = await readStoredDraftTrace();
  if (!trace) throw new Error("No DraftTrace has been stored yet.");
  const raw = JSON.stringify(trace, null, 2);
  try {
    await navigator.clipboard.writeText(raw);
  } catch (_) {
    const textarea = document.createElement("textarea");
    textarea.value = raw;
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.documentElement.appendChild(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  setStatus("DraftTrace copied");
}

function text(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function normalizedContactName(value) {
  return text(value).toLowerCase();
}

function elementText(node) {
  return text(node ? node.textContent || node.getAttribute?.("aria-label") || "" : "");
}

function simpleHash(value) {
  let hash = 2166136261;
  const raw = String(value || "");
  for (let index = 0; index < raw.length; index += 1) {
    hash ^= raw.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function newRequestId() {
  const random = typeof crypto !== "undefined" && crypto.getRandomValues ? crypto.getRandomValues(new Uint32Array(1))[0].toString(36) : Math.random().toString(36).slice(2);
  return `wa_req_${Date.now().toString(36)}_${random}`;
}

function threadIdFor(contact, messages) {
  const stableContact = text(contact) || "unknown";
  const anchor = messages
    .slice(0, 12)
    .map((message) => `${message.speaker}:${message.text}`)
    .join("|");
  return `wa_${simpleHash(`${stableContact}|${anchor}`)}`;
}

function runtimeRawMessage(error) {
  let raw = "";
  try {
    if (typeof error === "string") {
      raw = error;
    } else if (error && typeof error.message === "string") {
      raw = error.message;
    } else {
      raw = String(error || "");
    }
  } catch (_) {
    raw = "Extension context invalidated";
  }
  return raw;
}

function runtimeUnavailableMessage(error) {
  const raw = runtimeRawMessage(error);
  if (/extension was reloaded|extension context invalidated|extension runtime is unavailable/i.test(raw)) {
    return "Extension was reloaded. Refresh WhatsApp Web to reconnect Phone Copilot.";
  }
  return raw || "Extension runtime is unavailable. Refresh WhatsApp Web to reconnect Phone Copilot.";
}

function isExtensionInvalidation(error) {
  return /extension was reloaded|extension context invalidated|extension runtime is unavailable/i.test(runtimeRawMessage(error));
}

function markRuntimeUnavailable(error) {
  runtimeUnavailable = true;
  clearTimeout(autoTimer);
  if (observer) observer.disconnect();
  const message = runtimeUnavailableMessage(error);
  void storeDraftTrace(frontendFailureTrace({
    stage: "runtime_unavailable",
    error: message,
    controller_called: false,
    response_received: false,
  }));
  setStatus(message);
  return { ok: false, error: message };
}

function sendRuntimeMessage(type, payload = {}) {
  return new Promise((resolve) => {
    try {
      if (runtimeUnavailable || typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.id) {
        resolve(markRuntimeUnavailable("Extension runtime is unavailable."));
        return;
      }
      chrome.runtime.sendMessage({ type, payload }, (response) => {
        try {
          const runtimeError = chrome.runtime.lastError;
          if (runtimeError) {
            if (isExtensionInvalidation(runtimeError)) {
              resolve(markRuntimeUnavailable(runtimeError));
              return;
            }
            resolve({ ok: false, error: runtimeError.message });
            return;
          }
          resolve(response || { ok: false, error: "empty extension response" });
        } catch (error) {
          resolve(markRuntimeUnavailable(error));
        }
      });
    } catch (error) {
      resolve(markRuntimeUnavailable(error));
    }
  });
}

function sendRuntimePort(type, payload = {}) {
  return new Promise((resolve) => {
    let port = null;
    let settled = false;
    let timeout = null;
    let keepalive = null;
    const finish = (response) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      if (keepalive) clearInterval(keepalive);
      try {
        if (port) port.disconnect();
      } catch (_) {
        // Chrome may have already closed the port.
      }
      resolve(response);
    };
    try {
      if (runtimeUnavailable || typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.id) {
        finish(markRuntimeUnavailable("Extension runtime is unavailable."));
        return;
      }
      if (typeof chrome.runtime.connect !== "function") {
        sendRuntimeMessage(type, payload).then(finish);
        return;
      }
      const id = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
      port = chrome.runtime.connect({ name: LONG_REQUEST_PORT });
      timeout = setTimeout(() => {
        finish({ ok: false, error: "Phone Copilot background request timed out before the controller replied." });
      }, LONG_REQUEST_TIMEOUT_MS);
      port.onMessage.addListener((response) => {
        if (response?.pong) return;
        if (response && response.id && response.id !== id) return;
        const { id: _id, ...payloadResponse } = response || {};
        finish(payloadResponse || { ok: false, error: "empty extension response" });
      });
      port.onDisconnect.addListener(() => {
        if (settled) return;
        const runtimeError = chrome.runtime.lastError;
        if (runtimeError && isExtensionInvalidation(runtimeError)) {
          finish(markRuntimeUnavailable(runtimeError));
          return;
        }
        finish({
          ok: false,
          error: runtimeError?.message || "Phone Copilot background channel closed before the controller replied. Refresh WhatsApp Web and retry.",
        });
      });
      keepalive = setInterval(() => {
        try {
          port.postMessage({ id: `${id}_ping_${Date.now()}`, type: "pc:ping" });
        } catch (error) {
          finish({
            ok: false,
            error: error.message || "Phone Copilot background channel closed during keepalive.",
          });
        }
      }, LONG_REQUEST_KEEPALIVE_MS);
      port.postMessage({ id, type, payload });
    } catch (error) {
      finish(markRuntimeUnavailable(error));
    }
  });
}

function sendRuntime(type, payload = {}) {
  return sendRuntimePort(type, payload);
}

window.addEventListener("unhandledrejection", (event) => {
  if (!isExtensionInvalidation(event.reason)) return;
  event.preventDefault();
  markRuntimeUnavailable(event.reason);
});

window.addEventListener("error", (event) => {
  if (!isExtensionInvalidation(event.error || event.message)) return;
  event.preventDefault();
  markRuntimeUnavailable(event.error || event.message);
});

function isContactNameCandidate(value) {
  if (!value || value.length > 80) return false;
  if (/online|typing|last seen/i.test(value)) return false;
  if (/^(profile details|default-contact-refreshed|ic-|wa-|menu|search|video call)$/i.test(value)) return false;
  if (/^ic-[a-z-]+/i.test(value)) return false;
  return true;
}

function contactName() {
  const header = document.querySelector("#main header") || document.querySelector("header");
  if (!header) return "";
  const preferredSelectors = [
    "[data-testid='conversation-info-header-chat-title']",
    "[data-testid='conversation-info-header'] span[dir='auto']",
    "[data-testid='conversation-info-header'] [title]",
    "[data-testid='conversation-info-header']",
  ];
  for (const selector of preferredSelectors) {
    const candidate = header.querySelector(selector);
    const value = text(candidate?.getAttribute?.("title") || candidate?.textContent || "");
    if (isContactNameCandidate(value)) return value;
  }
  const candidates = Array.from(header.querySelectorAll('span[dir="auto"], [title], [data-testid] span'));
  for (const candidate of candidates) {
    const value = text(candidate.getAttribute?.("title") || candidate.textContent || "");
    if (isContactNameCandidate(value)) return value;
  }
  return "";
}

function messageText(container) {
  const spans = Array.from(container.querySelectorAll("span.selectable-text, span[dir='ltr'], span[dir='auto']"));
  const values = spans.map((span) => text(span.textContent)).filter(Boolean);
  const unique = [];
  for (const value of values) {
    if (!unique.includes(value)) unique.push(value);
  }
  return text(unique.join(" "));
}

function messageTimestamp(container) {
  const source = container.closest?.("[data-pre-plain-text]") ||
    container.querySelector?.("[data-pre-plain-text]") ||
    container;
  const value = source.getAttribute?.("data-pre-plain-text") || "";
  const match = value.match(/^\[([^\]]+)\]/);
  return match ? match[1] : "";
}

function timestampMillis(value) {
  const clean = text(value).replace(/\u202f|\xa0/g, " ");
  if (!clean) return Number.NaN;
  const iso = Date.parse(clean.replace(/Z$/, "+00:00"));
  if (Number.isFinite(iso)) return iso;
  let match = clean.match(/(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am|pm),?\s*(\d{4})-(\d{1,2})-(\d{1,2})/i);
  if (match) {
    let hour = Number(match[1]);
    const minute = Number(match[2]);
    const ampm = match[3].toLowerCase();
    if (ampm.startsWith("p") && hour < 12) hour += 12;
    if (ampm.startsWith("a") && hour === 12) hour = 0;
    return new Date(Number(match[4]), Number(match[5]) - 1, Number(match[6]), hour, minute).getTime();
  }
  match = clean.match(/(\d{1,2})\/(\d{1,2})\/(\d{4}),?\s*(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?/i);
  if (match) {
    let hour = Number(match[4]);
    const minute = Number(match[5]);
    const ampm = (match[6] || "").toLowerCase();
    if (ampm.startsWith("p") && hour < 12) hour += 12;
    if (ampm.startsWith("a") && hour === 12) hour = 0;
    return new Date(Number(match[3]), Number(match[2]) - 1, Number(match[1]), hour, minute).getTime();
  }
  return Number.NaN;
}

function collectMessages() {
  const rawContainers = [
    ...document.querySelectorAll("#main [data-pre-plain-text], #main .message-in, #main .message-out"),
  ];
  const seen = new Set();
  const messages = [];
  for (const container of rawContainers) {
    const bubble = container.closest(".message-in, .message-out") || container.closest("[data-pre-plain-text]") || container;
    if (seen.has(bubble)) continue;
    seen.add(bubble);
    const body = messageText(bubble);
    if (!body || body.length > 2000) continue;
    const speaker = bubble.closest(".message-out") || bubble.classList.contains("message-out") ? "me" : "other";
    messages.push({ speaker, text: body, timestamp: messageTimestamp(bubble) });
  }
  return messages;
}

function mergeMessages(existing, incoming) {
  const merged = new Map();
  for (const message of [...existing, ...incoming]) {
    const key = `${message.speaker}|${message.timestamp || ""}|${message.text}`;
    const order = Number.isFinite(message._pcOrder) ? message._pcOrder : messageOrderCounter++;
    const previous = merged.get(key);
    merged.set(key, {
      speaker: message.speaker,
      text: message.text,
      timestamp: message.timestamp || "",
      message_id: message.message_id || `wa_msg_${simpleHash(key)}`,
      _pcOrder: previous ? Math.max(previous._pcOrder, order) : order,
    });
  }
  return [...merged.values()]
    .sort((left, right) => {
      const leftTime = timestampMillis(left.timestamp);
      const rightTime = timestampMillis(right.timestamp);
      if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
        return leftTime - rightTime;
      }
      return left._pcOrder - right._pcOrder;
    })
    .slice(-420);
}

function publicMessages(messages) {
  return messages.map((message) => ({
    speaker: message.speaker,
    text: message.text,
    timestamp: message.timestamp || "",
    client_order: Number.isFinite(message._pcOrder) ? message._pcOrder : null,
    message_id: message.message_id || `wa_msg_${simpleHash(`${message.speaker}|${message.timestamp || ""}|${message.text}`)}`,
  }));
}

function refreshVisibleThreadCache() {
  const activeContact = contactName();
  if (activeContact !== cachedContactName) {
    cachedContactName = activeContact;
    cachedThreadMessages = [];
    lastAutoMessageKey = "";
  }
  const beforeKey = messageKey(publicMessages(cachedThreadMessages));
  cachedThreadMessages = mergeMessages(cachedThreadMessages, collectMessages());
  const afterKey = messageKey(publicMessages(cachedThreadMessages));
  if (beforeKey !== afterKey) setCount(cachedThreadMessages.length);
  return beforeKey !== afterKey;
}

function scrollPane() {
  const candidates = [
    document.querySelector("#main .copyable-area"),
    document.querySelector("#main [role='application']"),
    document.querySelector("#main div[tabindex='-1']"),
    document.querySelector("#main"),
    ...Array.from(document.querySelectorAll("#main div"))
      .filter((node) => node.scrollHeight > node.clientHeight + 100)
      .sort((a, b) => b.scrollHeight - a.scrollHeight),
  ].filter(Boolean);
  return candidates.find((node) => node.scrollHeight > node.clientHeight + 100) || null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout(promise, ms, label) {
  let timeout = null;
  try {
    return await Promise.race([
      promise,
      new Promise((resolve) => {
        timeout = setTimeout(() => resolve({ timedOut: true, label }), ms);
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

async function collectThreadMessages({ target = THREAD_CONTEXT_TARGET, deep = true } = {}) {
  const activeContact = contactName();
  if (activeContact !== cachedContactName) {
    cachedContactName = activeContact;
    cachedThreadMessages = [];
    lastAutoMessageKey = "";
    setCount(0);
  }
  const initialVisibleMessages = collectMessages();
  cachedThreadMessages = mergeMessages(cachedThreadMessages, initialVisibleMessages);
  if (!deep || cachedThreadMessages.length >= target) return cachedThreadMessages;

  const pane = scrollPane();
  if (!pane) return cachedThreadMessages;
  const startedNearBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 500;
  const originalTop = pane.scrollTop;
  let lastHeight = pane.scrollHeight;
  let stagnantPasses = 0;

  for (let pass = 0; pass < 36 && cachedThreadMessages.length < target; pass += 1) {
    pane.scrollTop = 0;
    setStatus(`collecting context ${cachedThreadMessages.length}/${target}`);
    await sleep(450);
    cachedThreadMessages = mergeMessages(cachedThreadMessages, collectMessages());
    if (pane.scrollHeight === lastHeight) {
      stagnantPasses += 1;
    } else {
      stagnantPasses = 0;
      lastHeight = pane.scrollHeight;
    }
    if (stagnantPasses >= 4) break;
  }

  pane.scrollTop = startedNearBottom ? pane.scrollHeight : originalTop;
  await sleep(150);
  cachedThreadMessages = mergeMessages(cachedThreadMessages, initialVisibleMessages);
  cachedThreadMessages = mergeMessages(cachedThreadMessages, collectMessages());
  return cachedThreadMessages;
}

function messageKey(messages) {
  return messages
    .slice(-8)
    .map((message) => `${message.speaker}:${message.text}`)
    .join("|");
}

function clickCandidate(node) {
  const target = node?.closest?.("button, [role='button'], [role='row'], [tabindex]") || node;
  if (!target) return false;
  target.scrollIntoView?.({ block: "center", inline: "nearest" });
  target.focus?.();
  const rect = target.getBoundingClientRect?.();
  const clientX = rect ? rect.left + rect.width / 2 : 0;
  const clientY = rect ? rect.top + rect.height / 2 : 0;
  const eventInit = { bubbles: true, cancelable: true, view: window, clientX, clientY };
  if (typeof PointerEvent === "function") {
    target.dispatchEvent(new PointerEvent("pointerdown", { ...eventInit, pointerId: 1, pointerType: "mouse", isPrimary: true }));
  }
  target.dispatchEvent(new MouseEvent("mousedown", eventInit));
  target.dispatchEvent(new MouseEvent("mouseup", eventInit));
  if (typeof PointerEvent === "function") {
    target.dispatchEvent(new PointerEvent("pointerup", { ...eventInit, pointerId: 1, pointerType: "mouse", isPrimary: true }));
  }
  target.click();
  target.dispatchEvent(new MouseEvent("dblclick", eventInit));
  return true;
}

function clickChatRow(row) {
  const targets = [
    row?.querySelector?.("a[href]"),
    row?.querySelector?.("button"),
    row?.querySelector?.("span[title]"),
    row?.querySelector?.("span[dir='auto']"),
    row?.querySelector?.("[data-testid='cell-frame-title']"),
    row?.querySelector?.("[data-testid='cell-frame-primary']"),
    row?.querySelector?.("[data-testid='cell-frame-secondary']"),
    row?.querySelector?.("[data-testid='cell-frame-container']"),
    row?.querySelector?.("[role='gridcell']"),
    row?.querySelector?.("[tabindex='0']"),
    row?.querySelector?.("[role='button']"),
    row,
  ].filter(Boolean);
  let clicked = false;
  for (const target of targets) {
    clicked = clickCandidate(target) || clicked;
  }
  const focusTarget = targets.find((target) => typeof target.focus === "function") || row;
  focusTarget?.focus?.();
  focusTarget?.dispatchEvent?.(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true, cancelable: true }));
  focusTarget?.dispatchEvent?.(new KeyboardEvent("keyup", { key: "Enter", code: "Enter", bubbles: true, cancelable: true }));
  return clicked;
}

async function trustedClickElement(node) {
  const target = node?.matches?.("[data-testid^='list-item-'], [data-testid='cell-frame-container'], [role='row'], [role='listitem']")
    ? node
    : node?.querySelector?.("[data-testid^='list-item-'], [data-testid='cell-frame-container'], [role='row'], [role='listitem'], [role='gridcell'], span[title], span[dir='auto']") || node;
  if (!target?.getBoundingClientRect) return { ok: false, error: "trusted click target missing" };
  target.scrollIntoView?.({ block: "center", inline: "nearest" });
  await sleep(120);
  const rect = target.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) {
    return { ok: false, error: "trusted click target not visible" };
  }
  return sendRuntime("pc:trusted-click", {
    x: Math.round(rect.left + Math.min(Math.max(rect.width * 0.22, 72), rect.width - 12)),
    y: Math.round(rect.top + rect.height / 2),
  });
}

function isVisibleElement(node) {
  if (!node?.getBoundingClientRect) return false;
  const rect = node.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 && rect.top < window.innerHeight && rect.left < window.innerWidth;
}

function archivePageHeading() {
  const selectors = [
    "h1",
    "h2",
    "[role='heading']",
    "header span[title]",
    "header span[dir='auto']",
    "header div[title]",
    "header div[dir='auto']",
  ];
  return Array.from(document.querySelectorAll(selectors.join(", ")))
    .find((node) => {
      if (!isVisibleElement(node)) return false;
      const label = text(node.getAttribute("title") || node.textContent || "");
      if (label !== "Archived") return false;
      const rect = node.getBoundingClientRect();
      return rect.top < Math.max(180, window.innerHeight * 0.25);
    }) || null;
}

function scoreArchiveRoot(node, heading = null) {
  if (!isVisibleElement(node)) return null;
  const label = text(`${node.getAttribute("aria-label") || ""} ${node.textContent || ""}`);
  const hasArchiveHelper = /These chats stay archived/i.test(label);
  const containsHeading = Boolean(heading && node.contains(heading));
  if (!hasArchiveHelper && !containsHeading) return null;
  const rect = node.getBoundingClientRect();
  if (rect.width < 260 || rect.height < 120) return null;
  const rowCount = archiveRowCandidates(node).length + archiveUnreadRowsFromBadges(node).length;
  if (!hasArchiveHelper && rowCount < 1) return null;
  return {
    node,
    rowCount,
    helperScore: hasArchiveHelper ? 1 : 0,
    headingScore: containsHeading ? 1 : 0,
    textLength: elementText(node).length,
  };
}

function archiveDrawerRoot(heading = archivePageHeading()) {
  const helperNodes = Array.from(document.querySelectorAll("p, span, div"))
    .filter((node) => {
      if (!isVisibleElement(node)) return false;
      const label = text(node.textContent || "");
      return /^These chats stay archived\b/i.test(label);
    });
  const seeds = [heading, ...helperNodes].filter(Boolean);
  const candidates = new Set();
  for (const seed of seeds) {
    for (let node = seed; node; node = node.parentElement) {
      candidates.add(node);
      if (node === document.body) break;
    }
  }
  const scored = Array.from(candidates)
    .map((node) => {
      if (!isVisibleElement(node)) return null;
      const rect = node.getBoundingClientRect();
      if (rect.width < 260 || rect.height < 160) return null;
      if (window.innerWidth >= 900 && rect.width > Math.min(760, window.innerWidth * 0.55)) return null;
      const label = text(`${node.getAttribute("aria-label") || ""} ${node.textContent || ""}`);
      const hasArchiveHelper = /These chats stay archived/i.test(label);
      const containsHeading = Boolean(heading && node.contains(heading));
      if (!hasArchiveHelper && !containsHeading) return null;
      const rowCount = archiveRowCandidates(node).length + archiveUnreadRowsFromBadges(node).length;
      if (rowCount < 1) return null;
      return {
        node,
        drawerScore: node.getAttribute("data-testid") === "drawer-left" ? 1 : 0,
        helperScore: hasArchiveHelper ? 1 : 0,
        headingScore: containsHeading ? 1 : 0,
        rowCount,
        textLength: elementText(node).length,
      };
    })
    .filter(Boolean)
    .sort((left, right) => (
      right.drawerScore - left.drawerScore ||
      right.helperScore - left.helperScore ||
      right.headingScore - left.headingScore ||
      right.rowCount - left.rowCount ||
      left.textLength - right.textLength
    ));
  return scored[0]?.node || null;
}

function archiveListRoot() {
  const explicit = Array.from(document.querySelectorAll("[data-testid='archived-chatlist']"))
    .find((node) => isVisibleElement(node));
  if (explicit) return explicit;
  const heading = archivePageHeading();
  const drawer = archiveDrawerRoot(heading);
  if (drawer) return drawer;
  const broadCandidates = Array.from(document.querySelectorAll("section, main, [role='application'], [role='grid'], [role='list'], [tabindex], div"));
  const ancestorCandidates = [];
  for (let node = heading; node; node = node.parentElement) {
    ancestorCandidates.push(node);
    if (node === document.body) break;
  }
  const roots = [...ancestorCandidates, ...broadCandidates]
    .map((node) => scoreArchiveRoot(node, heading))
    .filter(Boolean)
    .sort((left, right) => (
      right.helperScore - left.helperScore ||
      right.headingScore - left.headingScore ||
      right.rowCount - left.rowCount ||
      left.textLength - right.textLength
    ));
  return roots[0]?.node || null;
}

function mainListRoot() {
  const archiveRoot = archiveListRoot();
  const candidates = Array.from(document.querySelectorAll("[data-testid='chat-list'], [aria-label='Chat list'], #pane-side, [role='grid'], [role='list']"))
    .filter((node) => {
      if (!isVisibleElement(node)) return false;
      if (archiveRoot && (node === archiveRoot || node.contains(archiveRoot) || archiveRoot.contains(node))) return false;
      const label = text(`${node.getAttribute("aria-label") || ""} ${node.textContent || ""}`);
      if (/These chats stay archived/i.test(label)) return false;
      const rect = node.getBoundingClientRect();
      if (rect.width < 260 || rect.height < 120) return false;
      return true;
    })
    .sort((left, right) => {
      const leftRows = chatRowsInRoot(left).length;
      const rightRows = chatRowsInRoot(right).length;
      return rightRows - leftRows || elementText(left).length - elementText(right).length;
    });
  return candidates[0] || null;
}

function archiveEntry() {
  const candidates = Array.from(document.querySelectorAll("[role='button'], [role='row'], button, div"))
    .filter((node) => {
      const label = text(node.getAttribute("aria-label") || node.textContent || "");
      return /^archived\b/i.test(label) || /\barchived chats?\b/i.test(label);
    });
  return candidates
    .sort((left, right) => elementText(left).length - elementText(right).length)[0] || null;
}

async function openArchiveView() {
  if (archiveListRoot()) return true;
  const entry = archiveEntry();
  if (!entry) {
    setStatus("archive row not found");
    return false;
  }
  const trustedClick = await withTimeout(trustedClickElement(entry), 2500, "trusted_archive_entry_click");
  if (!trustedClick.ok) clickCandidate(entry);
  for (let pass = 0; pass < 10; pass += 1) {
    await sleep(350);
    if (archiveListRoot()) return true;
  }
  setStatus("archive page did not open");
  return false;
}

function chatRowsInRoot(root) {
  if (!root) return [];
  const rows = [...archiveUnreadRowsFromBadges(root), ...archiveRowCandidates(root)];
  const unique = [];
  const seen = new Set();
  for (const row of rows) {
    const frame = row.matches("[data-testid^='list-item-'], [data-testid='cell-frame-container'], [role='row'], [role='listitem']") ? row : row.closest("[data-testid^='list-item-'], [data-testid='cell-frame-container'], [role='row'], [role='listitem']") || row;
    if (seen.has(frame)) continue;
    seen.add(frame);
    if (frame === root || frame.id === "pane-side" || !isVisibleElement(frame)) continue;
    const rect = frame.getBoundingClientRect();
    if (rect.width < 180 || rect.height < 44 || rect.height > 180) continue;
    const label = text(`${frame.getAttribute("aria-label") || ""} ${frame.textContent || ""}`);
    if (!isArchiveChatRowLabel(label)) continue;
    unique.push(frame);
  }
  return unique;
}

function chatRows() {
  const root = archiveListRoot();
  return chatRowsInRoot(root);
}

function archiveRowCandidates(root) {
  if (!root?.querySelectorAll) return [];
  return Array.from(root.querySelectorAll("[data-testid^='list-item-'], [data-testid='cell-frame-container'], [role='row'], [role='listitem'], [role='button'], [tabindex='0'], [tabindex='-1'], div"))
    .filter((row) => {
      if (!isVisibleElement(row)) return false;
      const rect = row.getBoundingClientRect();
      if (rect.width < 220 || rect.height < 44 || rect.height > 180) return false;
      const label = text(`${row.getAttribute("aria-label") || ""} ${row.textContent || ""}`);
      if (/These chats stay archived/i.test(label)) return false;
      if (/\bArchived\b/i.test(label) && label.length < 80) return false;
      return isArchiveChatRowLabel(label);
    });
}

function archiveUnreadRowsFromBadges(root) {
  if (!root?.querySelectorAll) return [];
  const rows = [];
  const seen = new Set();
  for (const badge of archiveUnreadBadgeNodes(root)) {
    const row = rowFromUnreadBadge(badge, root);
    if (!row || seen.has(row)) continue;
    seen.add(row);
    rows.push(row);
  }
  return rows;
}

function archiveUnreadBadgeNodes(root) {
  if (!root?.querySelectorAll) return [];
  const rootRect = root.getBoundingClientRect?.();
  return Array.from(root.querySelectorAll("span, div"))
    .filter((node) => {
      const value = text(node.textContent || "");
      if (!/^\d{1,3}$/.test(value)) return false;
      if (!isVisibleElement(node)) return false;
      const rect = node.getBoundingClientRect();
      if (rect.width < 10 || rect.height < 10 || rect.width > 54 || rect.height > 54) return false;
      if (rootRect && rect.left < rootRect.left + rootRect.width * 0.55) return false;
      const style = window.getComputedStyle?.(node);
      const colorBlob = `${style?.backgroundColor || ""} ${style?.color || ""}`;
      return /rgb\((?:\s*37|\s*0|\s*18|\s*34|\s*20|\s*48|\s*30)[,\s]+(?:\s*211|\s*168|\s*186|\s*197|\s*208|\s*199|\s*215)/i.test(colorBlob) ||
        Boolean(node.closest("[aria-label*='unread' i], [data-icon='unread-count']")) ||
        value.length <= 3;
    });
}

function rowFromUnreadBadge(badge, root) {
  let node = badge;
  while (node && node !== root && node.parentElement) {
    node = node.parentElement;
    if (!isVisibleElement(node)) continue;
    const rect = node.getBoundingClientRect();
    if (rect.width < 220 || rect.height < 44 || rect.height > 180) continue;
    const label = text(`${node.getAttribute("aria-label") || ""} ${node.textContent || ""}`);
    if (!isArchiveChatRowLabel(label)) continue;
    if (/These chats stay archived/i.test(label)) continue;
    return node;
  }
  return null;
}

function isArchiveChatRowLabel(label) {
  if (!label) return false;
  if (/^archived\b/i.test(label)) return false;
  if (/^archive-refreshed/i.test(label)) return false;
  if (/^chat list\b/i.test(label)) return false;
  if (/^(all|unread|favorites|groups)\d*$/i.test(label)) return false;
  if (/^\d+\s+unread chats?$/i.test(label)) return false;
  if (label.length < 3) return false;
  return true;
}

function unreadChatRowsInRoot(root) {
  const badgeRows = root ? archiveUnreadRowsFromBadges(root) : [];
  const rows = [...badgeRows, ...chatRowsInRoot(root)];
  const seen = new Set();
  const actionableRoot = rootRowsAreActionable(root);
  return rows.filter((row) => {
    if (seen.has(row)) return false;
    seen.add(row);
    const label = text(`${row.getAttribute("aria-label") || ""} ${row.textContent || ""}`);
    if (!isArchiveChatRowLabel(label)) return false;
    if (actionableRoot && root !== archiveListRoot()) return true;
    if (/\b(unread|new message|new messages)\b/i.test(label)) return true;
    if (row.querySelector("[aria-label*='unread' i], [data-icon='unread-count'], span[aria-label*='unread' i]")) return true;
    return hasUnreadCountBadge(row);
  });
}

function rootRowsAreActionable(root) {
  if (!root) return false;
  const archiveRoot = archiveListRoot();
  if (archiveRoot && root === archiveRoot) return true;
  if (root !== mainListRoot()) return false;
  return Array.from(document.querySelectorAll("[role='tab'], button"))
    .some((node) => {
      if (!isVisibleElement(node)) return false;
      const label = text(`${node.getAttribute("aria-selected") || ""} ${node.textContent || ""}`);
      return /^true\s*Unread\b/i.test(label) || /^Unread\s*\d*/i.test(label);
    });
}

function unreadChatRows() {
  return unreadChatRowsInRoot(archiveListRoot());
}

function hasUnreadCountBadge(row) {
  const rowRect = row.getBoundingClientRect?.();
  if (!rowRect) return false;
  return Array.from(row.querySelectorAll("span, div"))
    .some((node) => {
      const value = text(node.textContent || "");
      if (!/^\d{1,3}$/.test(value)) return false;
      if (!isVisibleElement(node)) return false;
      const rect = node.getBoundingClientRect();
      if (rect.width > 44 || rect.height > 44 || rect.width < 10 || rect.height < 10) return false;
      return rect.left > rowRect.left + rowRect.width * 0.62;
    });
}

function rowKey(row) {
  return text(row.getAttribute("aria-label") || row.textContent || "").slice(0, 220);
}

function rowContactName(row) {
  const selectors = [
    "span[title]",
    "[data-testid='cell-frame-title']",
    "[data-testid='cell-frame-primary']",
    "span[dir='auto']",
  ];
  for (const selector of selectors) {
    const node = row?.querySelector?.(selector);
    const value = text(node?.getAttribute?.("title") || node?.textContent || "");
    if (value && isArchiveChatRowLabel(value)) return value;
  }
  const label = rowKey(row);
  const firstLine = text(label.split(/\n/)[0] || label);
  return firstLine;
}

function archiveRowAllowed(row) {
  const allowed = archiveAllowedContactNames.map(normalizedContactName).filter(Boolean);
  if (allowed.includes("*")) return true;
  if (!allowed.length) return false;
  const name = normalizedContactName(rowContactName(row));
  if (!name) return false;
  return allowed.includes(name);
}

function chatOpenState() {
  const visibleMessages = publicMessages(collectMessages());
  const currentContact = contactName();
  return {
    contact: currentContact,
    messages: visibleMessages,
    visibleKey: messageKey(visibleMessages),
    threadId: threadIdFor(currentContact, visibleMessages),
  };
}

function chatOpenedAfterClick(previous, key) {
  const current = chatOpenState();
  if (!current.contact && !current.messages.length) return null;
  if (current.messages.length && current.visibleKey !== previous.visibleKey) return current;
  if (current.contact && current.contact !== previous.contact) return current;
  if (current.threadId !== previous.threadId) return current;
  if (current.contact && key.includes(current.contact)) return current;
  if (composer() && current.messages.length) return current;
  return null;
}

async function openUnreadChatRow(row, scope = "chat") {
  const previous = chatOpenState();
  const key = rowKey(row);
  if (!isArchiveChatRowLabel(key)) {
    return { opened: false, key, reason: "not_a_chat_row" };
  }
  const trustedClick = await withTimeout(trustedClickElement(row), 2500, "trusted_click");
  if (!trustedClick.ok) {
    pcLog(`${scope} trusted click failed`, { key, error: trustedClick.timedOut ? trustedClick.label : trustedClick.error || "" });
    if (!clickChatRow(row)) {
      return { opened: false, key, reason: "row_click_failed" };
    }
  }
  for (let pass = 0; pass < 32; pass += 1) {
    await sleep(250);
    const current = chatOpenedAfterClick(previous, key);
    if (current) return { opened: true, key, contact: current.contact, threadId: current.threadId };
  }
  return { opened: false, key, reason: "chat_open_not_confirmed" };
}

async function openArchivedChatRow(row) {
  return openUnreadChatRow(row, "archive");
}

function composer() {
  const selectors = [
    "footer [contenteditable='true'][role='textbox']",
    "footer div[contenteditable='true']",
    "[data-testid='conversation-compose-box-input']",
    "div[contenteditable='true'][role='textbox']",
  ];
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (node) return node;
  }
  return null;
}

function sendButton() {
  const selectors = [
    "button[aria-label='Send']",
    "button span[data-icon='send']",
    "span[data-icon='send']",
  ];
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (!node) continue;
    return node.closest("button") || node;
  }
  return null;
}

function composerEchoDuplicated(actual, expected) {
  const actualText = normalizeBubble(actual);
  const expectedText = normalizeBubble(expected);
  return Boolean(expectedText) && (
    actualText === `${expectedText}${expectedText}` ||
    actualText === `${expectedText} ${expectedText}`
  );
}

function insertIntoComposer(reply) {
  const box = composer();
  if (!box) throw new Error("Could not find WhatsApp composer.");
  box.focus();
  document.execCommand("selectAll", false);
  document.execCommand("delete", false);
  document.execCommand("insertText", false, reply);
  box.dispatchEvent(new Event("input", { bubbles: true }));
  const inserted = composerText();
  if (composerEchoDuplicated(inserted, reply)) {
    document.execCommand("selectAll", false);
    document.execCommand("delete", false);
    document.execCommand("insertText", false, reply);
    box.dispatchEvent(new Event("input", { bubbles: true }));
  }
  if (composerEchoDuplicated(composerText(), reply)) {
    throw new Error("Stopped send queue because composer duplicated the bubble.");
  }
}

function clickSend() {
  const button = sendButton();
  if (!button) throw new Error("Could not find WhatsApp send button.");
  button.click();
}

function textareaSequence(panel) {
  const raw = panel.querySelector("[data-pc='reply']").value || "";
  const parts = raw.split(/\n+|\s+\/\s+/).map((part) => text(part)).filter(Boolean);
  return parts.length ? parts : lastDraftSequence;
}

function normalizeBubble(value) {
  return text(value).toLowerCase().replace(/[^\w\s]/g, "").trim();
}

function bubbleSimilarity(left, right) {
  const leftWords = normalizeBubble(left).split(/\s+/).filter(Boolean);
  const rightWords = normalizeBubble(right).split(/\s+/).filter(Boolean);
  if (!leftWords.length || !rightWords.length) return 0;
  const leftPairs = new Set(leftWords.slice(0, -1).map((word, index) => `${word} ${leftWords[index + 1]}`));
  const rightPairs = new Set(rightWords.slice(0, -1).map((word, index) => `${word} ${rightWords[index + 1]}`));
  if (!leftPairs.size || !rightPairs.size) {
    return leftWords[0] === rightWords[0] ? 1 : 0;
  }
  let overlap = 0;
  for (const pair of leftPairs) {
    if (rightPairs.has(pair)) overlap += 1;
  }
  return (2 * overlap) / (leftPairs.size + rightPairs.size);
}

function queueSafety(sequence) {
  const bubbles = sequence.map((part) => text(part)).filter(Boolean);
  const duplicates = [];
  const nearDuplicates = [];
  const seen = new Set();
  for (let index = 0; index < bubbles.length; index += 1) {
    const normalized = normalizeBubble(bubbles[index]);
    if (seen.has(normalized)) duplicates.push(bubbles[index]);
    seen.add(normalized);
    for (let right = index + 1; right < bubbles.length; right += 1) {
      if (bubbleSimilarity(bubbles[index], bubbles[right]) >= 0.86) {
        nearDuplicates.push([bubbles[index], bubbles[right]]);
      }
    }
  }
  const joined = normalizeBubble(bubbles.join(" "));
  const contradictions = [];
  if (joined.includes("you failed") && (joined.includes("you passed") || joined.includes("in shaa allah you passed"))) {
    contradictions.push("pass_fail_contradiction");
  }
  return {
    passed: bubbles.length > 0 && !duplicates.length && !nearDuplicates.length && !contradictions.length,
    bubbles,
    duplicates,
    nearDuplicates,
    contradictions,
  };
}

function composerText() {
  const box = composer();
  return text(box ? box.textContent || box.innerText || "" : "");
}

function clearComposerIfPreviousDraft() {
  const box = composer();
  if (!box || !lastDraftSequence.length) return;
  const current = normalizeBubble(composerText());
  const previous = normalizeBubble(lastDraftSequence.join(" "));
  if (!current || current !== previous) return;
  box.focus();
  document.execCommand("selectAll", false);
  document.execCommand("delete", false);
  box.dispatchEvent(new Event("input", { bubbles: true }));
}

function latestIncomingKey() {
  const messages = publicMessages(collectMessages());
  return messages
    .filter((message) => message.speaker !== "me")
    .slice(-6)
    .map((message) => message.text)
    .join("|");
}

function latestVisibleSpeaker() {
  const messages = publicMessages(collectMessages());
  return messages.length ? messages[messages.length - 1].speaker : "";
}

function randomAutoSendDelayMs() {
  return 1000 + Math.floor(Math.random() * 2001);
}

function assertQueueCanContinue({ baselineIncomingKey, expectedComposerText = "" } = {}) {
  if (baselineIncomingKey && latestIncomingKey() !== baselineIncomingKey && latestVisibleSpeaker() !== "me") {
    throw new Error("Stopped send queue because a new incoming message appeared.");
  }
  const currentComposerText = composerText();
  if (currentComposerText && currentComposerText !== expectedComposerText) {
    throw new Error("Stopped send queue because the composer was edited.");
  }
}

async function sendSequence(sequence, options = {}) {
  const { interruptible = false, autoSend = false, baselineIncomingKey = "" } = options;
  const safety = queueSafety(sequence);
  if (!safety.bubbles.length) throw new Error("No draft to send.");
  if (!safety.passed) throw new Error("Stopped unsafe send queue.");
  const startingIncomingKey = baselineIncomingKey || latestIncomingKey();
  const bubbles = safety.bubbles;
  for (const bubble of bubbles) {
    if (interruptible) assertQueueCanContinue({ baselineIncomingKey: startingIncomingKey });
    insertIntoComposer(bubble);
    await sleep(autoSend ? randomAutoSendDelayMs() : 120);
    if (interruptible) assertQueueCanContinue({ baselineIncomingKey: startingIncomingKey, expectedComposerText: bubble });
    clickSend();
    await sleep(autoSend ? randomAutoSendDelayMs() : 350);
    if (interruptible) assertQueueCanContinue({ baselineIncomingKey: startingIncomingKey });
  }
}

function ensurePanel() {
  let panel = document.getElementById(PANEL_ID);
  if (panel) return panel;
  panel = document.createElement("section");
  panel.id = PANEL_ID;
  panel.innerHTML = `
    <style>
      #${PANEL_ID} {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 999999;
        width: 360px;
        font: 13px/1.35 Arial, sans-serif;
        color: #e9edef;
        background: #111b21;
        border: 1px solid #2a3942;
        box-shadow: 0 12px 40px rgba(0,0,0,.35);
        border-radius: 10px;
        overflow: hidden;
      }
      #${PANEL_ID} header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 12px;
        background: #202c33;
        font-weight: 700;
      }
      #${PANEL_ID} .pc-body { padding: 12px; display: grid; gap: 10px; }
      #${PANEL_ID} label { display: grid; gap: 4px; color: #b8c4ca; }
      #${PANEL_ID} select,
      #${PANEL_ID} input {
        background: #0b141a;
        color: #e9edef;
        border: 1px solid #2a3942;
        border-radius: 6px;
        padding: 7px;
      }
      #${PANEL_ID} textarea {
        width: 100%;
        min-height: 86px;
        resize: vertical;
        box-sizing: border-box;
        background: #0b141a;
        color: #e9edef;
        border: 1px solid #2a3942;
        border-radius: 6px;
        padding: 8px;
      }
      #${PANEL_ID} .pc-row { display: flex; gap: 8px; }
      #${PANEL_ID} button {
        flex: 1;
        border: 0;
        border-radius: 6px;
        padding: 8px;
        color: #062018;
        background: #00a884;
        font-weight: 700;
        cursor: pointer;
      }
      #${PANEL_ID} button.secondary { background: #8696a0; color: #111b21; }
      #${PANEL_ID} button.danger { background: #ffb4a9; color: #32130f; }
      #${PANEL_ID} .pc-status { color: #b8c4ca; min-height: 18px; }
      #${PANEL_ID} .pc-count { color: #7dd8c1; font-size: 12px; min-height: 16px; }
      #${PANEL_ID} .pc-debug { color: #90a4ae; font-size: 11px; min-height: 14px; overflow-wrap: anywhere; }
      #${PANEL_ID} .pc-memory { border-top: 1px solid #2a3942; padding-top: 10px; display: grid; gap: 8px; }
      #${PANEL_ID} .pc-memory textarea { min-height: 54px; }
      #${PANEL_ID}.collapsed .pc-body { display: none; }
    </style>
    <header>
      <span>Phone Copilot</span>
      <button class="secondary" data-pc="toggle" style="max-width:42px;padding:4px;">-</button>
    </header>
    <div class="pc-body">
      <label>Mode
        <select data-pc="mode">
          <option value="review">review</option>
          <option value="auto-review">auto-review</option>
          <option value="auto-send">auto-send</option>
          <option value="autonomous">autonomous archive</option>
        </select>
      </label>
      <label>Relationship
        <select data-pc="relationship">
          <option value="unknown">unknown</option>
          <option value="close_friend">close_friend</option>
          <option value="casual_friend">casual_friend</option>
          <option value="romantic_interest">romantic_interest</option>
          <option value="family">family</option>
          <option value="professional">professional</option>
          <option value="university">university</option>
        </select>
      </label>
      <label>Tone
        <select data-pc="tone">
          <option value="bolder">bolder</option>
          <option value="normal">normal</option>
          <option value="riskier">riskier</option>
        </select>
      </label>
      <label>Controller
        <input data-pc="endpoint" value="http://127.0.0.1:8765">
      </label>
      <div class="pc-row">
        <button data-pc="draft">Draft</button>
        <button data-pc="collect" class="secondary">Collect thread</button>
      </div>
      <div class="pc-row">
        <button data-pc="scan-archive" class="secondary">Scan archive</button>
        <button data-pc="reload-page" class="secondary">Reload WA</button>
      </div>
      <div class="pc-count" data-pc="count">context 0/220</div>
      <div class="pc-debug" data-pc="debug-version">content ${CONTENT_VERSION}</div>
      <div class="pc-debug" data-pc="trace-preview">trace empty</div>
      <div class="pc-row">
        <button data-pc="insert" class="secondary">Insert</button>
        <button data-pc="send" class="danger">Send</button>
      </div>
      <div class="pc-row">
        <button data-pc="copy-trace" class="secondary">Copy last DraftTrace</button>
      </div>
      <textarea data-pc="reply" placeholder="Draft appears here"></textarea>
      <div class="pc-memory">
        <label>Question
          <input data-pc="question" placeholder="What should I remember?">
        </label>
        <label>Answer / fact
          <textarea data-pc="answer" placeholder="Save a fact, plan, or tone note for this chat"></textarea>
        </label>
        <button data-pc="save-memory" class="secondary">Save memory</button>
      </div>
      <div class="pc-status" data-pc="status">${STATUS_IDLE}</div>
    </div>
  `;
  document.documentElement.appendChild(panel);
  bindPanel(panel);
  return panel;
}

function setStatus(message) {
  const panel = ensurePanel();
  const node = panel.querySelector("[data-pc='status']");
  if (node) node.textContent = message;
}

function setReply(reply, sequence = []) {
  lastDraft = reply || "";
  lastDraftSequence = Array.isArray(sequence) && sequence.length ? sequence.map((item) => text(item)).filter(Boolean) : (lastDraft ? [lastDraft] : []);
  const panel = ensurePanel();
  const node = panel.querySelector("[data-pc='reply']");
  if (node) node.value = lastDraftSequence.length > 1 ? lastDraftSequence.join("\n") : lastDraft;
}

function setCount(count, target = THREAD_CONTEXT_TARGET) {
  const panel = ensurePanel();
  const node = panel.querySelector("[data-pc='count']");
  if (node) node.textContent = `context ${count}/${target}`;
}

function setQuestion(question) {
  const panel = ensurePanel();
  const node = panel.querySelector("[data-pc='question']");
  if (node && question && !node.value) node.value = question;
}

async function loadSettings(panel) {
  const response = await sendRuntime("pc:get-settings");
  if (!response.ok) {
    setStatus(response.error || "could not load settings");
    return;
  }
  const settings = response.settings || {};
  archiveAllowedContactNames = Array.isArray(settings.archiveAllowedContactNames) ? settings.archiveAllowedContactNames : [];
  panel.querySelector("[data-pc='mode']").value = settings.mode || "review";
  panel.querySelector("[data-pc='relationship']").value = settings.relationshipType || "unknown";
  panel.querySelector("[data-pc='tone']").value = settings.toneRisk || "bolder";
  panel.querySelector("[data-pc='endpoint']").value = settings.endpoint || "http://127.0.0.1:8765";
  scheduleArchiveScan();
}

async function saveSettings(panel) {
  const payload = {
    mode: panel.querySelector("[data-pc='mode']").value,
    relationshipType: panel.querySelector("[data-pc='relationship']").value,
    toneRisk: panel.querySelector("[data-pc='tone']").value,
    endpoint: panel.querySelector("[data-pc='endpoint']").value,
  };
  await sendRuntime("pc:save-settings", payload);
  return payload;
}

async function draftNow(source = "manual") {
  let draftOutcome = { status: "failed", message: "draft did not complete" };
  let requestId = "";
  let requestThreadId = "";
  let activeContact = "";
  let controllerCalled = false;
  let responseReceived = false;
  let draftBaselineIncomingKey = "";
  let trace = null;
  const startedAt = new Date().toISOString();
  const buildTrace = (updates = {}) => {
    const previous = trace || {};
    const next = {
      content_version: CONTENT_VERSION,
      created_at: previous.created_at || startedAt,
      source,
      request_id: requestId,
      thread_id: requestThreadId,
      chat_name: activeContact,
      stage: "frontend_failure",
      error: "",
      controller_called: controllerCalled,
      response_received: responseReceived,
      controller_response_received: responseReceived,
      ui_rendered: false,
      active_request_id: activeDraftRequestId,
      draft_in_flight: draftInFlight,
      latest_burst_preview: draftBaselineIncomingKey,
      response_accepted_or_rejected: "pending",
      rejection_reason: "",
      rendered_draft_units_count: 0,
      visible_draft_text_length: 0,
      ...previous,
      ...updates,
    };
    next.request_id = updates.request_id || previous.request_id || requestId;
    next.thread_id = updates.thread_id || previous.thread_id || requestThreadId;
    next.chat_name = updates.chat_name || previous.chat_name || activeContact;
    next.controller_called = updates.controller_called ?? controllerCalled;
    next.response_received = updates.response_received ?? responseReceived;
    next.controller_response_received = updates.controller_response_received ?? responseReceived;
    next.active_request_id = updates.active_request_id || activeDraftRequestId;
    next.draft_in_flight = updates.draft_in_flight ?? draftInFlight;
    next.latest_burst_preview = updates.latest_burst_preview || previous.latest_burst_preview || draftBaselineIncomingKey;
    return next;
  };
  const storeTrace = async (updates = {}) => {
    trace = buildTrace(updates);
    await storeDraftTrace(trace);
    return trace;
  };
  const finishDraft = (message, status = "failed", extra = {}) => {
    draftOutcome = { status, message, ...extra };
    setStatus(message);
    pcLog("final status", message);
    return draftOutcome;
  };
  const logSkippedController = (reason) => {
    pcLog("request sent", { skipped: true, reason });
    pcLog("response received", { skipped: true, reason });
  };
  pcLog("draft started", { source, draft_in_flight: draftInFlight });
  if (draftInFlight) {
    await storeSuppressedDraftTrace(source);
    logSkippedController("draft_already_running");
    pcLog("render rejected", { reason: "draft_already_running" });
    finishDraft("draft already running", "skipped");
    return;
  }
  draftInFlight = true;
  clearTimeout(autoTimer);
  autoTimer = null;
  const panel = ensurePanel();
  try {
    const settings = await saveSettings(panel);
    const previousDraftSequence = lastDraftSequence.slice();
    clearComposerIfPreviousDraft();
    setReply("", []);
    const collected = await collectThreadMessages({ target: THREAD_CONTEXT_TARGET, deep: true });
    const messages = publicMessages(collected);
    pcLog("collected messages", {
      count: messages.length,
      latest: messages.slice(-3).map((message) => `${message.speaker}:${message.text}`),
    });
    setCount(messages.length);
    if (!messages.length) {
      await storeTrace({
        stage: "frontend_failure",
        error: "no visible WhatsApp messages found",
        response_accepted_or_rejected: "rejected",
        rejection_reason: "no_visible_messages",
      });
      logSkippedController("no_visible_messages");
      pcLog("render rejected", { reason: "no_visible_messages" });
      finishDraft("no visible WhatsApp messages found", "failed");
      return;
    }
    draftBaselineIncomingKey = messages
      .filter((message) => message.speaker !== "me")
      .slice(-6)
      .map((message) => message.text)
      .join("|");
    const autoSource = source === "auto" || source.endsWith("-auto");
    if (autoSource) {
      if (messages[messages.length - 1].speaker === "me") {
        await storeTrace({
          stage: "waiting_for_incoming",
          error: "latest visible message is from self",
          response_accepted_or_rejected: "rejected",
          rejection_reason: "latest_message_from_self",
        });
        logSkippedController("latest_message_from_self");
        pcLog("render rejected", { reason: "latest_message_from_self" });
        finishDraft("waiting for incoming message", "skipped");
        return;
      }
      const key = messageKey(messages);
      if (key === lastAutoMessageKey) {
        await storeTrace({
          stage: "duplicate_auto_message_key",
          error: "auto draft already handled this visible tail",
          response_accepted_or_rejected: "rejected",
          rejection_reason: "duplicate_auto_message_key",
        });
        logSkippedController("duplicate_auto_message_key");
        pcLog("render rejected", { reason: "duplicate_auto_message_key" });
        finishDraft("waiting for new incoming message", "skipped");
        return;
      }
      lastAutoMessageKey = key;
    }
    requestId = newRequestId();
    activeDraftRequestId = requestId;
    setStatus(autoSource ? "auto drafting..." : "drafting...");
    activeContact = contactName();
    if (!activeContact) {
      await storeTrace({
        stage: "frontend_failure",
        error: "could not identify WhatsApp chat name",
        response_accepted_or_rejected: "rejected",
        rejection_reason: "missing_chat_name",
      });
      logSkippedController("missing_chat_name");
      pcLog("render rejected", { reason: "missing_chat_name" });
      finishDraft("could not identify WhatsApp chat name", "failed");
      return;
    }
    requestThreadId = threadIdFor(activeContact, messages);
    await storeTrace({
      stage: "request_started",
      draft_request_started: true,
      request_id: requestId,
      thread_id: requestThreadId,
      chat_name: activeContact,
      latest_burst_preview: draftBaselineIncomingKey,
      draft_in_flight: draftInFlight,
      controller_called: false,
      response_received: false,
      controller_response_received: false,
      response_accepted_or_rejected: "pending",
      rejection_reason: "",
      rendered_draft_units_count: 0,
      visible_draft_text_length: 0,
    });
    controllerCalled = true;
    pcLog("request sent", { request_id: requestId, thread_id: requestThreadId, chat_name: activeContact });
    const response = await sendRuntime("pc:draft", {
      requestId,
      threadId: requestThreadId,
      contactName: activeContact,
      messages,
      avoidCandidates: previousDraftSequence,
      source,
    });
    responseReceived = true;
    pcLog("response received", {
      request_id: requestId,
      ok: Boolean(response.ok),
      error: response.error || "",
    });
    if (requestId !== activeDraftRequestId) {
      await storeTrace({
        stage: "stale_request_id",
        controller_response_received: true,
        response_received: true,
        response_accepted_or_rejected: "rejected",
        rejection_reason: "stale_request_id",
      });
      pcLog("render rejected", { reason: "stale_request_id" });
      finishDraft("ignored stale draft response", "skipped");
      return;
    }
    if (!response.ok) {
      await storeTrace({
        stage: "frontend_failure",
        error: response.error || "draft failed",
        controller_response_received: true,
        response_received: true,
        response_accepted_or_rejected: "rejected",
        rejection_reason: response.error || "draft_failed",
      });
      pcLog("render rejected", { reason: response.error || "draft_failed" });
      finishDraft(response.error || "draft failed", "failed");
      return;
    }
    const data = response.data || {};
    if (data.request_id && data.request_id !== requestId) {
      await storeTrace({
        ...(data.draft_trace || {}),
        stage: "mismatched_response_request_id",
        controller_response_received: true,
        response_received: true,
        response_accepted_or_rejected: "rejected",
        rejection_reason: "mismatched_response_request_id",
      });
      pcLog("render rejected", { reason: "mismatched_response_request_id" });
      finishDraft("ignored mismatched draft response", "skipped");
      return;
    }
    const currentContact = contactName();
    const currentThreadId = threadIdFor(currentContact, publicMessages(cachedThreadMessages));
    if (currentContact !== activeContact || (data.thread_id && data.thread_id !== requestThreadId) || currentThreadId !== requestThreadId) {
      await storeTrace({
        ...(data.draft_trace || {}),
        stage: "thread_changed_before_render",
        controller_response_received: true,
        response_received: true,
        response_accepted_or_rejected: "rejected",
        rejection_reason: "thread_changed_before_render",
        response_thread_id: data.thread_id || "",
        current_thread_id: currentThreadId,
        current_chat_name: currentContact,
      });
      pcLog("render rejected", { reason: "thread_changed_before_render" });
      finishDraft("ignored draft for previous chat", "skipped");
      return;
    }
    setReply(data.reply || "", data.reply_sequence || []);
    await storeTrace({
      ...(data.draft_trace || {}),
      stage: data.reply ? "rendered" : "no_draft",
      controller_response_received: true,
      response_received: true,
      response_accepted_or_rejected: data.reply ? "accepted" : "rejected",
      rejection_reason: data.reply ? "" : (data.auto_send_blocked_reason || data.automation_decision || "no_draft_returned"),
      ui_rendered: Boolean(data.reply),
      ui_render_rejection_reason: data.reply ? "" : (data.auto_send_blocked_reason || data.automation_decision || "no_draft_returned"),
      rendered_draft_units_count: Array.isArray(data.reply_sequence) ? data.reply_sequence.length : (data.reply ? 1 : 0),
      visible_draft_text_length: (data.reply || "").length,
      final_reply_units: Array.isArray(data.reply_sequence) ? data.reply_sequence : (data.reply ? [data.reply] : []),
    });
    setCount(data.collected_message_count || messages.length, data.context_target_count || THREAD_CONTEXT_TARGET);
    if (data.relationship_inferred_from_thread && data.relationship_type) {
      const relationshipNode = panel.querySelector("[data-pc='relationship']");
      if (relationshipNode && relationshipNode.value === "unknown") {
        relationshipNode.value = data.relationship_type;
        await saveSettings(panel);
      }
    }
    if (Array.isArray(data.questions) && data.questions.length) setQuestion(data.questions[0]);
    if (!data.reply) {
      const reason = data.auto_send_blocked_reason || data.automation_decision || "no draft returned";
      pcLog("render rejected", { reason });
      if (reason === "latest_message_from_self") {
        finishDraft("latest visible message is yours - waiting for incoming", "skipped");
      } else {
        finishDraft(`no draft - ${reason}`, "failed");
      }
      return;
    }
    pcLog("render accepted", {
      request_id: requestId,
      units: Array.isArray(data.reply_sequence) ? data.reply_sequence.length : 1,
    });
    const contextStatus = data.context_minimum_met ? "context ok" : `context short ${data.collected_message_count || messages.length}/${data.context_target_count || THREAD_CONTEXT_TARGET}`;
    const burstPreview = Array.isArray(data.latest_incoming_burst) ? data.latest_incoming_burst.slice(-3).join(" | ").slice(0, 140) : "";
    const replyPreview = Array.isArray(data.reply_sequence) ? data.reply_sequence.slice(0, 2).join(" | ").slice(0, 100) : "";
    finishDraft(
      `${data.automation_decision || "REVIEW_REQUIRED"} - ${data.intent_type || "auto"} - ${contextStatus}${burstPreview ? ` - latest: ${burstPreview}` : ""}${replyPreview ? ` - reply: ${replyPreview}` : ""}`,
      "drafted",
      { requestId, threadId: requestThreadId, contact: activeContact, units: Array.isArray(data.reply_sequence) ? data.reply_sequence.length : 1 },
    );
    const sendAllowed = Boolean(data.auto_send_allowed || data.automation_decision === "SEND_ALLOWED");
    const archiveAutoSource = source === "archive-auto";
    if ((settings.mode === "auto-send" || settings.mode === "autonomous") && archiveAutoSource && sendAllowed && data.reply) {
      await sendSequence(data.reply_sequence || [data.reply], {
        interruptible: true,
        autoSend: true,
        baselineIncomingKey: draftBaselineIncomingKey,
      });
      await storeTrace({
        stage: "auto_send_completed",
        ui_rendered: true,
        auto_send_completed: true,
      });
      finishDraft(settings.mode === "autonomous" ? "sent by autonomous archive gate" : "sent by auto-send gate", "sent", {
        requestId,
        threadId: requestThreadId,
        contact: activeContact,
        units: Array.isArray(data.reply_sequence) ? data.reply_sequence.length : 1,
      });
    }
  } catch (error) {
    const message = error.message || String(error);
    if (!controllerCalled) pcLog("request sent", { skipped: true, reason: message });
    if (!responseReceived) pcLog("response received", { skipped: true, reason: message });
    await storeTrace({
      stage: "frontend_failure",
      error: message,
      response_accepted_or_rejected: "rejected",
      rejection_reason: message,
      ui_rendered: false,
    });
    pcLog("render rejected", { reason: message });
    finishDraft(message, "failed");
  } finally {
    draftInFlight = false;
    return draftOutcome;
  }
}

function isScanKeyStalled(scopedKey) {
  const until = stalledScanKeys.get(scopedKey) || 0;
  if (!until) return false;
  if (Date.now() < until) return true;
  stalledScanKeys.delete(scopedKey);
  return false;
}

function markScanKeyStalled(scopedKey) {
  stalledScanKeys.set(scopedKey, Date.now() + STALLED_SCAN_BACKOFF_MS);
}

function archiveScanKey(row) {
  return normalizedContactName(rowContactName(row)) || rowKey(row);
}

function isControllerTimeoutResult(result) {
  const message = text(result?.message || "");
  return /timed out|draft_request|controller timed out/i.test(message);
}

async function scanArchiveNow({ autonomous = false } = {}) {
  if (archiveScanRunning) return;
  archiveScanRunning = true;
  try {
    setStatus(autonomous ? "autonomous archive scan..." : "scanning archive...");
    let acted = 0;
    let skipped = 0;
    let unreadTotal = 0;
    const attemptedThisScan = new Set();
    while (true) {
      if (autonomous && ensurePanel().querySelector("[data-pc='mode']")?.value !== "autonomous") {
        setStatus("archive scan stopped - autonomous mode off");
        break;
      }
      const opened = await withTimeout(openArchiveView(), 15000, "open_archive_view");
      if (opened?.timedOut) {
        skipped += 1;
        setStatus(`archive scan stopped - ${opened.label} timed out`);
        break;
      }
      if (!opened) break;
      const unreadRows = unreadChatRows();
      unreadTotal = Math.max(unreadTotal, unreadRows.length);
      if (!unreadRows.length) break;
      const allowedRows = unreadRows.filter((candidate) => archiveRowAllowed(candidate));
      const blockedRows = unreadRows.length - allowedRows.length;
      if (!allowedRows.length) {
        setStatus(blockedRows ? `archive scan blocked ${blockedRows} row(s) outside allowlist` : "archive scan blocked - allowlist empty");
        break;
      }
      const row = allowedRows.find((candidate) => {
        const key = archiveScanKey(candidate);
        return key && !archiveProcessedKeys.has(key) && !attemptedThisScan.has(key) && !isScanKeyStalled(key);
      });
      if (!row) break;
      const key = archiveScanKey(row);
      const rowLabel = rowKey(row);
      attemptedThisScan.add(key);
      const openResult = await withTimeout(openArchivedChatRow(row), 12000, "open_archived_chat");
      if (openResult?.timedOut) {
        skipped += 1;
        pcLog("archive row open failed", { key, row: rowLabel, reason: openResult.label || "timed_out" });
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} drafted, skipped ${skipped}`);
        continue;
      }
      if (!openResult.opened) {
        skipped += 1;
        pcLog("archive row open failed", { key, row: rowLabel, reason: openResult.reason || "" });
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} drafted, skipped ${skipped}`);
        continue;
      }
      cachedThreadMessages = [];
      cachedContactName = "";
      lastAutoMessageKey = "";
      const result = await withTimeout(draftNow(autonomous ? "archive-auto" : "manual"), LONG_REQUEST_TIMEOUT_MS + 10000, "draft_request");
      if (result?.timedOut) {
        markScanKeyStalled(key);
        skipped += 1;
        pcLog("archive draft not completed", { key, row: rowLabel, status: "timed_out", message: result.label || "" });
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} drafted, skipped ${skipped}`);
        continue;
      }
      if (isControllerTimeoutResult(result)) {
        markScanKeyStalled(key);
        skipped += 1;
        pcLog("archive draft not completed", { key, row: rowLabel, status: result?.status || "timeout", message: result?.message || "" });
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} drafted, skipped ${skipped}`);
        continue;
      }
      if (result && (result.status === "drafted" || result.status === "sent")) {
        archiveProcessedKeys.add(key);
        acted += 1;
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} unread chats${skipped ? `, skipped ${skipped}` : ""}`);
      } else {
        skipped += 1;
        pcLog("archive draft not completed", { key, row: rowLabel, status: result?.status || "unknown", message: result?.message || "" });
        setStatus(`archive scan progress ${acted}/${Math.max(unreadTotal, attemptedThisScan.size)} drafted, skipped ${skipped}`);
      }
      await sleep(1200);
    }
    if (!unreadTotal) {
      setStatus("archive scan found no unread chats");
      return;
    }
    setStatus(`archive scan drafted ${acted}/${unreadTotal} unread chats${skipped ? `, skipped ${skipped}` : ""}`);
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    archiveScanRunning = false;
  }
}

async function collectNow() {
  const messages = await collectThreadMessages({ target: THREAD_CONTEXT_TARGET, deep: true });
  setCount(messages.length);
  setStatus(messages.length >= THREAD_CONTEXT_TARGET ? "collected deep thread context" : `only found ${messages.length} messages`);
}

async function saveMemoryNow() {
  const panel = ensurePanel();
  await saveSettings(panel);
  const question = text(panel.querySelector("[data-pc='question']").value);
  const answer = text(panel.querySelector("[data-pc='answer']").value);
  if (!question || !answer) {
    setStatus("question and answer required");
    return;
  }
  const messages = publicMessages(await collectThreadMessages({ target: THREAD_CONTEXT_TARGET, deep: false }));
  const response = await sendRuntime("pc:save-memory", {
    contactName: contactName(),
    question,
    answer,
    messages,
  });
  if (!response.ok) {
    setStatus(response.error || "memory save failed");
    return;
  }
  panel.querySelector("[data-pc='answer']").value = "";
  setStatus("memory saved");
}

function scheduleAutoDraft() {
  if (draftInFlight || runtimeUnavailable) {
    clearTimeout(autoTimer);
    autoTimer = null;
    return;
  }
  const panel = ensurePanel();
  const mode = panel.querySelector("[data-pc='mode']").value;
  if (mode !== "autonomous") {
    clearTimeout(autoTimer);
    autoTimer = null;
    return;
  }
  clearTimeout(autoTimer);
  autoTimer = null;
}

function scheduleArchiveScan() {
  const panel = document.getElementById(PANEL_ID);
  const mode = panel?.querySelector("[data-pc='mode']")?.value || "review";
  clearInterval(archiveScanTimer);
  archiveScanTimer = null;
  if (mode !== "autonomous" || runtimeUnavailable) return;
  archiveScanTimer = setInterval(() => scanArchiveNow({ autonomous: true }), ARCHIVE_SCAN_INTERVAL_MS);
  clearTimeout(autoTimer);
  autoTimer = setTimeout(() => scanArchiveNow({ autonomous: true }), 1200);
}

function bindPanel(panel) {
  panel.addEventListener("click", async (event) => {
    const action = event.target?.getAttribute?.("data-pc");
    try {
      if (action === "toggle") panel.classList.toggle("collapsed");
      if (action === "draft") await draftNow("manual");
      if (action === "collect") await collectNow();
      if (action === "scan-archive") await scanArchiveNow({ autonomous: panel.querySelector("[data-pc='mode']")?.value === "autonomous" });
      if (action === "reload-page") window.location.reload();
      if (action === "save-memory") await saveMemoryNow();
      if (action === "copy-trace") await copyLastDraftTrace();
      if (action === "insert") {
        const sequence = textareaSequence(panel);
        insertIntoComposer(sequence.join("\n"));
        setStatus("inserted");
      }
      if (action === "send") {
        await sendSequence(textareaSequence(panel));
        setStatus("sent by user click");
      }
    } catch (error) {
      setStatus(error.message || String(error));
    }
  });
  panel.addEventListener("change", async () => {
    await saveSettings(panel);
    scheduleArchiveScan();
  });
  panel.addEventListener("blur", () => saveSettings(panel), true);
  loadSettings(panel);
}

function startObserver() {
  if (observer) return;
  observer = new MutationObserver(() => {
    refreshVisibleThreadCache();
    scheduleAutoDraft();
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

installPageDebugBridge();
ensurePanel();
void readStoredDraftTrace().then((trace) => {
  if (trace) void storeDraftTrace(trace);
});
startObserver();
