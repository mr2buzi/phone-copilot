(() => {
  const DEFAULT_VERSION = "wa-trace-bridge-20260605-archive-only-allow-all-004012";
  const script = document.currentScript;
  const VERSION = script?.dataset?.phoneCopilotVersion || DEFAULT_VERSION;
  if (window.__phoneCopilotDebugBridgeReady && window.__phoneCopilotDebugBridgeVersion === VERSION) return;

  let lastTrace = null;
  try {
    const existing = Object.getOwnPropertyDescriptor(window, "__phoneCopilotLastDraftTrace");
    if (existing?.get) lastTrace = existing.get.call(window);
  } catch (_) {
    lastTrace = null;
  }

  const defineGetter = (name, getter) => {
    Object.defineProperty(window, name, {
      configurable: true,
      enumerable: false,
      get: getter,
    });
  };

  defineGetter("__phoneCopilotDebugBridgeReady", () => true);
  defineGetter("__phoneCopilotDebugBridgeVersion", () => VERSION);
  defineGetter("__phoneCopilotContentVersion", () => VERSION);
  defineGetter("__phoneCopilotLastDraftTrace", () => lastTrace);

  if (document.documentElement) {
    document.documentElement.dataset.phoneCopilotDebugBridgeVersion = VERSION;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data || {};
    if (data.source !== "phone-copilot-content" || data.type !== "draft-trace") return;
    lastTrace = data.trace || null;
    console.log("[PhoneCopilot] bridge trace stored", {
      request_id: lastTrace && lastTrace.request_id,
      stage: lastTrace && lastTrace.stage,
      version: VERSION,
    });
  });

  console.log("[PhoneCopilot] page debug bridge loaded", VERSION);
})();
