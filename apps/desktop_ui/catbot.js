const catbotUrlParams = new URLSearchParams(window.location.search);
const catbotDebounceOverride = (catbotUrlParams.has("live250") || catbotUrlParams.has("live550"))
  ? Number(catbotUrlParams.get("debounce_ms"))
  : NaN;

const catbotState = {
  messages: [],
  context: [],
  busy: false,
  threadId: localStorage.getItem("catbotThreadId") || "",
  pendingMessages: [],
  pendingTimer: null,
  pendingStartedAt: 0,
  debounceMs: Number.isFinite(catbotDebounceOverride)
    ? Math.min(Math.max(catbotDebounceOverride, 250), 30000)
    : 25000,
};
window.catbotState = catbotState;
let catbotRouteRequestId = 0;

function setCatbotLastRoutePayload(payload) {
  window.catbotLastRoutePayload = payload || {};
  const element = document.getElementById("catbotLastRoutePayload");
  if (element) element.textContent = JSON.stringify(payload || {});
}

async function api(path, method = "GET", body = null) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (error) {
    const wrapped = new Error("Catbot request failed before the server replied");
    wrapped.status = "network";
    wrapped.cause = error;
    throw wrapped;
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || payload.error || `Catbot request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function renderMessages() {
  const container = document.getElementById("catbotMessages");
  const threadLabel = document.getElementById("catbotThreadId");
  const status = document.getElementById("catbotBatchStatus");
  if (threadLabel) threadLabel.textContent = `thread: ${catbotState.threadId || "new"}`;
  if (status) {
    if (catbotState.busy) {
      status.textContent = "thinking";
    } else if (catbotState.pendingMessages.length) {
      const count = catbotState.pendingMessages.length;
      status.textContent = `waiting for more (${count} message${count === 1 ? "" : "s"})`;
    } else {
      status.textContent = "";
    }
  }
  if (!catbotState.messages.length) {
    container.innerHTML = '<p class="catbot-empty">Start typing.</p>';
    return;
  }
  container.innerHTML = catbotState.messages.map((message, index) => `
    <div class="catbot-message-row ${message.role === "me" ? "catbot-row-me" : "catbot-row-bot"}">
      <div class="catbot-message">${escapeHtml(message.text)}</div>
      <div class="catbot-rating">
        ${message.sessionId ? `
          <button type="button" class="secondary compact-button ${message.rating === "thumbs_up" ? "selected" : ""}" data-rating="thumbs_up" data-index="${index}">👍</button>
          <button type="button" class="secondary compact-button ${message.rating === "thumbs_down" ? "selected" : ""}" data-rating="thumbs_down" data-index="${index}">👎</button>
        ` : ""}
      </div>
    </div>
  `).join("");
  container.querySelectorAll("[data-rating]").forEach((button) => {
    button.addEventListener("click", () => rateMessage(Number(button.dataset.index), button.dataset.rating));
  });
  container.scrollTop = container.scrollHeight;
}

function combinedPendingText(messages) {
  return messages.map((item) => item.text).filter(Boolean).join("\n");
}

function scheduleCatbotReply() {
  if (catbotState.pendingTimer) {
    clearTimeout(catbotState.pendingTimer);
  }
  renderMessages();
  catbotState.pendingTimer = setTimeout(() => {
    flushPendingMessages();
  }, catbotState.debounceMs);
}

async function flushPendingMessages() {
  if (catbotState.busy || !catbotState.pendingMessages.length) return;
  if (catbotState.pendingTimer) {
    clearTimeout(catbotState.pendingTimer);
    catbotState.pendingTimer = null;
  }
  const pending = catbotState.pendingMessages.splice(0);
  const text = combinedPendingText(pending);
  await sendMessageBatch(text, pending.map((item) => item.text));
}

function queueUserMessage(text) {
  catbotState.messages.push({ role: "catbot", text });
  catbotState.pendingMessages.push({ text, timestamp: Date.now() });
  scheduleCatbotReply();
}

async function sendMessageBatch(text, userParts) {
  if (catbotState.busy) return;
  catbotState.busy = true;
  const requestId = (catbotRouteRequestId += 1);
  setCatbotLastRoutePayload({ pending: true, request_id: requestId, incoming: text });
  renderMessages();
  try {
    const response = await api("/api/training/catbot/chat", "POST", {
      incoming: text,
      context: catbotState.context.slice(-12),
      contact_name: "Catbot",
      relationship_type: "romantic_interest",
      intent_type: "auto",
      diversity_mode: "natural",
      thread_id: catbotState.threadId,
    });
    setCatbotLastRoutePayload({ ok: true, request_id: requestId, incoming: text, response });
    if (response.thread_id) {
      catbotState.threadId = response.thread_id;
      localStorage.setItem("catbotThreadId", catbotState.threadId);
    }
    const replySequence = Array.isArray(response.reply_sequence) && response.reply_sequence.length
      ? response.reply_sequence
      : (Array.isArray(response.candidate?.sequence) && response.candidate.sequence.length ? response.candidate.sequence : [response.reply || ""]);
    replySequence.forEach((part, index) => {
      catbotState.messages.push({
        role: "me",
        text: part || "",
        sessionId: index === replySequence.length - 1 ? response.session_id : "",
        rating: "",
      });
    });
    catbotState.context.push(text, response.reply || "");
  } catch (error) {
    const status = error.status ? ` [${error.status}]` : "";
    setCatbotLastRoutePayload({
      ok: false,
      request_id: requestId,
      incoming: text,
      status: error.status || "",
      detail: error.message || "request failed",
    });
    catbotState.messages.push({ role: "catbot", text: `Catbot error${status}: ${error.message || "request failed"}`, sessionId: "", rating: "" });
  } finally {
    catbotState.busy = false;
    renderMessages();
    if (catbotState.pendingMessages.length) {
      scheduleCatbotReply();
    }
  }
}

async function rateMessage(index, rating) {
  const message = catbotState.messages[index];
  if (!message?.sessionId || !rating) return;
  message.rating = rating;
  renderMessages();
  try {
    await api("/api/training/catbot/feedback", "POST", {
      session_id: message.sessionId,
      rating,
    });
  } catch (error) {
    message.rating = "";
    message.text = `${message.text}\n${error.message}`;
    renderMessages();
  }
}

document.getElementById("catbotForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("catbotInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  queueUserMessage(text);
});

document.getElementById("catbotResetThread")?.addEventListener("click", () => {
  if (catbotState.pendingTimer) {
    clearTimeout(catbotState.pendingTimer);
    catbotState.pendingTimer = null;
  }
  catbotState.messages = [];
  catbotState.context = [];
  catbotState.pendingMessages = [];
  catbotState.busy = false;
  catbotState.threadId = "";
  localStorage.removeItem("catbotThreadId");
  renderMessages();
});

renderMessages();
