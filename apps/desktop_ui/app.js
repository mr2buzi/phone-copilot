const state = {
  selectedSuggestionIndex: 0,
  emergencyStop: false,
  automationState: {},
  targetCycleResult: null,
  latestPayload: null,
  pollInFlight: false,
  feedbackReason: "too formal",
};

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed." }));
    throw new Error(error.detail || error.error || "Request failed.");
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

function showBanner(message) {
  const banner = document.getElementById("haltBanner");
  const text = document.getElementById("haltBannerText");
  banner.classList.remove("hidden");
  text.textContent = message;
}

function clearBanner(force = false) {
  const payload = state.latestPayload;
  if (!force && payload?.halted) {
    return;
  }
  document.getElementById("haltBanner").classList.add("hidden");
  document.getElementById("haltBannerText").textContent = "";
}

function formatMs(value) {
  return `${Math.round(Number(value || 0))} ms`;
}

function renderMetrics(payload) {
  const metrics = payload?.last || {};
  document.getElementById("loopLatency").textContent = formatMs(metrics.loop_time_ms);
  document.getElementById("shotLatency").textContent = formatMs(metrics.screenshot_time_ms);
  document.getElementById("ocrLatency").textContent = formatMs(metrics.ocr_time_ms);
  document.getElementById("classificationLatency").textContent = formatMs(metrics.classification_time_ms);
  document.getElementById("plannerLatency").textContent = formatMs(metrics.planner_time_ms);
  document.getElementById("executionLatency").textContent = formatMs(metrics.execution_time_ms);
}

function renderSuggestions(items) {
  const container = document.getElementById("replySuggestions");
  container.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No reply suggestions are currently available.";
    container.appendChild(empty);
    return;
  }

  state.selectedSuggestionIndex = Math.min(state.selectedSuggestionIndex, items.length - 1);
  items.forEach((text, index) => {
    const card = document.createElement("button");
    card.className = `suggestion suggestion-card ${index === state.selectedSuggestionIndex ? "selected" : ""}`;
    card.type = "button";
    const bubbles = String(text).split(" / ").map((part) => `<span>${escapeHtml(part)}</span>`).join("");
    card.innerHTML = `
      <span class="suggestion-label">Draft ${index + 1}</span>
      <span class="suggestion-bubbles">${bubbles}</span>
    `;
    card.onclick = () => {
      state.selectedSuggestionIndex = index;
      renderSuggestions(items);
      renderReplyQuality(state.latestPayload || {});
      syncEditedReply(items[index] || "");
    };
    container.appendChild(card);
  });
  syncEditedReply(items[state.selectedSuggestionIndex] || "");
}

function syncEditedReply(text) {
  const input = document.getElementById("editedReplyInput");
  if (input && document.activeElement !== input) {
    input.value = text;
  }
}

function renderReplyQuality(payload) {
  const container = document.getElementById("replyQuality");
  container.innerHTML = "";
  const candidates = payload.draft_candidates || [];
  const selected = candidates[state.selectedSuggestionIndex] || candidates[0];
  const relationship = payload.relationship_type || selected?.relationship_type || "unknown";
  const intent = payload.intent_type || "other";
  const decision = payload.final_decision || selected?.final_decision || "review";
  const blocked = payload.blocked_reason || payload.auto_send_blocked_reason || selected?.blocked_reason || "None";
  const timings = payload.timings_ms || {};
  const retrieved = payload.retrieved_examples || selected?.retrieved_examples || [];

  const summary = document.createElement("div");
  summary.className = "quality-summary";
  summary.innerHTML = `
    <div><span>Relationship</span><strong>${escapeHtml(relationship)}</strong></div>
    <div><span>Intent</span><strong>${escapeHtml(intent)}</strong></div>
    <div><span>Retrieved examples</span><strong>${Number(payload.retrieved_examples_count || selected?.retrieved_examples_count || 0)}</strong></div>
    <div><span>Context messages</span><strong>${Number(payload.context_messages_count || 0)}</strong></div>
    <div><span>Scrolls</span><strong>${Number(payload.scrolls_performed || 0)}</strong></div>
    <div><span>Context stop</span><strong>${escapeHtml(payload.context_stopped_reason || "None")}</strong></div>
    <div><span>Decision</span><strong>${escapeHtml(decision)}</strong></div>
    <div><span>Blocked reason</span><strong>${escapeHtml(blocked)}</strong></div>
    <div><span>Retrieval backend</span><strong>${escapeHtml(payload.retrieval_backend || "lexical")}</strong></div>
    <div><span>Vector results</span><strong>${Number(payload.vector_results_count || 0)}</strong></div>
    <div><span>Fallback used</span><strong>${payload.lexical_fallback_used ? "Yes" : "No"}</strong></div>
    <div><span>Diversity</span><strong>${escapeHtml(payload.diversity_mode || selected?.diversity_mode || "natural")}</strong></div>
  `;
  container.appendChild(summary);

  if (Object.keys(timings).length) {
    const timingBox = document.createElement("div");
    timingBox.className = "debug-block";
    timingBox.innerHTML = `<strong>Timings</strong><pre>${escapeHtml(JSON.stringify(timings, null, 2))}</pre>`;
    container.appendChild(timingBox);
  }

  if (retrieved.length) {
    const retrievedBox = document.createElement("div");
    retrievedBox.className = "debug-block";
    retrievedBox.innerHTML = `
      <strong>Retrieved examples</strong>
      ${retrieved
        .map(
          (example) => `
            <div class="debug-example">
              <small>${escapeHtml(example.retrieval_id || "")} | ${escapeHtml(example.relationship_type || "")} | ${escapeHtml(example.intent_type || "")} | score ${Number(example.score || 0)}</small>
              <p>${escapeHtml(example.incoming || "")} -> ${escapeHtml(example.my_reply || "")}</p>
              <small>${escapeHtml(example.reason || "")}</small>
            </div>
          `,
        )
        .join("")}
    `;
    container.appendChild(retrievedBox);
  }

  if (payload.prompt_preview) {
    const promptBox = document.createElement("div");
    promptBox.className = "debug-block";
    promptBox.innerHTML = `<strong>Prompt preview</strong><pre>${escapeHtml(payload.prompt_preview)}</pre>`;
    container.appendChild(promptBox);
  }

  if (!candidates.length) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No scored candidates yet.";
    container.appendChild(empty);
    return;
  }

  candidates.forEach((candidate, index) => {
    const scores = candidate.critic_scores || {};
    const item = document.createElement("div");
    item.className = `quality-candidate ${index === state.selectedSuggestionIndex ? "selected" : ""}`;
    item.innerHTML = `
      <strong>${escapeHtml(candidate.candidate_kind || `candidate ${index + 1}`)}</strong>
      <p>${escapeHtml(candidate.text || "")}</p>
      <small>
        style ${Number(scores.style_match_score || 0)} |
        relevance ${Number(scores.relevance_score || 0)} |
        risk ${Number(scores.risk_score || 0)} |
        ${escapeHtml(candidate.final_decision || "review")}
      </small>
      ${candidate.blocked_reason ? `<small>${escapeHtml(candidate.blocked_reason)}</small>` : ""}
      ${candidate.why_this_matches ? `<small>${escapeHtml(candidate.why_this_matches)}</small>` : ""}
    `;
    container.appendChild(item);
  });
}

function renderVerificationErrors(items) {
  const container = document.getElementById("verificationErrors");
  container.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No recent verification errors.";
    container.appendChild(empty);
    return;
  }

  items.slice().reverse().forEach((reason) => {
    const item = document.createElement("div");
    item.className = "log-item";
    item.textContent = reason;
    container.appendChild(item);
  });
}

function renderUnreadQueue(items) {
  const container = document.getElementById("unreadQueue");
  container.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No queued threads.";
    container.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "queue-item";
    row.innerHTML = `
      <div class="queue-title-row">
        <strong>${escapeHtml(item.contact_name)}</strong>
        <span class="queue-badge ${item.unread ? "queue-badge-unread" : ""}">
          ${item.unread ? "Unread" : "Visible"}
        </span>
      </div>
      <small>${escapeHtml(item.preview || "No preview available.")}</small>
    `;

    const actions = document.createElement("div");
    actions.className = "inline-actions";

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "secondary compact-button";
    openButton.textContent = "Open";
    openButton.onclick = () => runAction("/api/automation/open-thread", { contact_name: item.contact_name });

    actions.appendChild(openButton);
    row.appendChild(actions);
    container.appendChild(row);
  });
}

function renderThreadContext(context, payload) {
  const container = document.getElementById("threadContext");
  container.innerHTML = "";

  if (!context) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "Open a thread and scroll for context to populate this panel.";
    container.appendChild(empty);
    return;
  }

  const header = document.createElement("div");
  header.className = "context-header";
  header.innerHTML = `
    <div>
      <strong>${escapeHtml(context.contact_name)}</strong>
      <p class="summary">Messages loaded: ${Number(context.message_count || 0)}</p>
    </div>
    <span class="queue-badge ${payload.blacklisted ? "queue-badge-blocked" : ""}">
      ${payload.blacklisted ? "Blacklisted" : "Allowed"}
    </span>
  `;
  container.appendChild(header);

  const messages = context.full_conversation || [];
  if (!messages.length) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No parsed thread messages are available yet.";
    container.appendChild(empty);
    return;
  }

  messages.slice(-12).forEach((entry) => {
    const line = document.createElement("div");
    line.className = `thread-line ${entry.speaker === "me" ? "speaker-me" : "speaker-other"}`;
    line.innerHTML = `
      <span class="thread-speaker">${entry.speaker === "me" ? "Me" : context.contact_name}</span>
      <span>${escapeHtml(entry.text)}</span>
    `;
    container.appendChild(line);
  });
}

function renderAutomationStatus(payload) {
  const container = document.getElementById("automationStatus");
  const automationState = state.automationState || {};
  const currentMode = automationState.mode || payload.automation_mode?.mode || "review";
  const confidence = Number(
    automationState.confidence_threshold ?? payload.automation_mode?.confidence_threshold ?? 0.85,
  );
  const blacklist = automationState.blacklist_contacts || [];
  const currentContact = automationState.current_contact || payload.thread_context?.contact_name || "None";
  const targetCycle = automationState.target_cycle || {};
  const targets = targetCycle.targets || [];
  const cycleResults = state.targetCycleResult?.results || [];

  container.innerHTML = `
    <div class="automation-panel">
      <div class="status-grid">
        <div>
          <label for="automationModeSelect">Mode</label>
          <select id="automationModeSelect">
            <option value="review" ${currentMode === "review" ? "selected" : ""}>Review</option>
            <option value="auto-draft" ${currentMode === "auto-draft" ? "selected" : ""}>Auto Draft</option>
            <option value="auto-send" ${currentMode === "auto-send" ? "selected" : ""}>Auto Send</option>
          </select>
        </div>
        <div>
          <label for="confidenceSlider">Confidence</label>
          <input id="confidenceSlider" type="range" min="0" max="100" value="${Math.round(confidence * 100)}" />
          <span id="confidenceValue" class="summary">${confidence.toFixed(2)}</span>
        </div>
      </div>

      <div class="button-row">
        <button id="scanInboxButton" type="button" class="secondary">Scan Inbox</button>
        <button id="processQueueButton" type="button" class="secondary">Process Next</button>
        <button id="scrollContextButton" type="button" class="secondary">Scroll Context</button>
        <button id="generateDraftButton" type="button" class="primary">Generate Draft</button>
        <button id="typeAiDraftButton" type="button" class="secondary">Type AI Draft</button>
        <button id="autoSendButton" type="button" class="secondary">Auto Send</button>
        <button id="dryRunTargetCycleButton" type="button" class="secondary">Dry-run Targets</button>
        <button id="runTargetCycleButton" type="button" class="secondary">Run Targets</button>
      </div>

      <div class="status-info">
        <div>Queue: ${Number(automationState.queue_size || 0)}</div>
        <div>Blacklist: ${Number(automationState.blacklist_count || 0)}</div>
        <div>Current thread: ${escapeHtml(currentContact)}</div>
        <div>Automation: ${automationState.automation_enabled ? "Enabled" : "Manual"}</div>
        <div>Target cycle: ${targetCycle.enabled ? "Enabled" : "Disabled"}</div>
        <div>Target mode: ${escapeHtml(targetCycle.mode || "review_only")}</div>
      </div>

      <div class="debug-block">
        <strong>Targets</strong>
        ${
          targets.length
            ? targets
                .map((target) => `<div class="queue-title-row"><span>${escapeHtml(target.name)}</span><small>${target.enabled ? "enabled" : "disabled"} | auto-send ${target.auto_send ? "on" : "off"}</small></div>`)
                .join("")
            : '<p class="summary">No allowlisted targets configured.</p>'
        }
      </div>

      <div class="debug-block">
        <strong>Target cycle results</strong>
        ${
          cycleResults.length
            ? cycleResults
                .map(
                  (result) => `
                    <div class="debug-example">
                      <small>${escapeHtml(result.contact || "")} | ${escapeHtml(result.status || "")} | ${escapeHtml(result.intent_type || "")} | ${escapeHtml(result.relationship_type || "")}</small>
                      <p>${escapeHtml(result.latest_incoming || "")} -> ${escapeHtml(result.selected_reply || "")}</p>
                      <small>${escapeHtml(result.reason || result.decision || "")}</small>
                    </div>
                  `,
                )
                .join("")
            : '<p class="summary">No target cycle run yet.</p>'
        }
      </div>

      <div class="input-row">
        <input id="manualThreadInput" class="text-input" type="text" placeholder="Open thread by contact name" />
        <button id="manualThreadButton" type="button" class="secondary">Open</button>
      </div>

      <div class="input-row">
        <input id="blacklistInput" class="text-input" type="text" placeholder="Blacklist contact name" />
        <button id="addBlacklistButton" type="button" class="secondary">Add</button>
      </div>

      <div id="blacklistList" class="pill-list">
        ${
          blacklist.length
            ? blacklist
                .map(
                  (name) => `
                    <button type="button" class="pill remove-pill" data-contact="${escapeHtml(name)}">
                      ${escapeHtml(name)} <span>Remove</span>
                    </button>
                  `,
                )
                .join("")
            : '<p class="summary">No blacklisted contacts.</p>'
        }
      </div>
    </div>
  `;

  const modeSelect = document.getElementById("automationModeSelect");
  const confidenceSlider = document.getElementById("confidenceSlider");
  const confidenceValue = document.getElementById("confidenceValue");
  const blacklistInput = document.getElementById("blacklistInput");
  const manualThreadInput = document.getElementById("manualThreadInput");

  modeSelect.onchange = () => setAutomationMode(modeSelect.value, Number(confidenceSlider.value) / 100);
  confidenceSlider.oninput = () => {
    confidenceValue.textContent = (Number(confidenceSlider.value) / 100).toFixed(2);
  };
  confidenceSlider.onchange = () => setAutomationMode(modeSelect.value, Number(confidenceSlider.value) / 100);

  document.getElementById("scanInboxButton").onclick = () => runAction("/api/automation/scan-inbox");
  document.getElementById("processQueueButton").onclick = () => runAction("/api/automation/process-queue");
  document.getElementById("scrollContextButton").onclick = () => runAction("/api/automation/scroll-for-context");
  document.getElementById("generateDraftButton").onclick = () => runAction("/api/automation/auto-generate-draft");
  document.getElementById("typeAiDraftButton").onclick = () =>
    runAction("/api/approve/ai-draft-to-compose", { suggestion_index: state.selectedSuggestionIndex });
  document.getElementById("autoSendButton").onclick = () => runAction("/api/automation/auto-send-reply");
  document.getElementById("dryRunTargetCycleButton").onclick = () => runTargetCycle(true);
  document.getElementById("runTargetCycleButton").onclick = () => runTargetCycle(false);
  document.getElementById("manualThreadButton").onclick = () => {
    const name = manualThreadInput.value.trim();
    if (!name) {
      showBanner("Enter a contact name to open a thread.");
      return;
    }
    runAction("/api/automation/open-thread", { contact_name: name });
  };
  document.getElementById("addBlacklistButton").onclick = () => {
    const contact = blacklistInput.value.trim();
    if (!contact) {
      showBanner("Enter a contact name to blacklist.");
      return;
    }
    runStatusAction("/api/automation/blacklist/add", {
      action: "add",
      contact_name: contact,
    }).then(() => {
      blacklistInput.value = "";
    });
  };

  container.querySelectorAll(".remove-pill").forEach((button) => {
    button.addEventListener("click", () => {
      runStatusAction("/api/automation/blacklist/remove", {
        action: "remove",
        contact_name: button.dataset.contact || "",
      });
    });
  });
}

async function runTargetCycle(dryRun) {
  try {
    state.targetCycleResult = await api(`/api/automation/run-target-cycle?dry_run=${dryRun ? "true" : "false"}`, "POST");
    const automationState = await api("/api/automation/state").catch(() => ({}));
    state.automationState = automationState || {};
    renderAutomationStatus(state.latestPayload || {});
  } catch (error) {
    showBanner(error.message);
  }
}

function updateButtons(payload) {
  const approve = document.getElementById("approveTypeButton");
  const sendDraft = document.getElementById("sendDraftButton");
  const reject = document.getElementById("rejectButton");
  const stop = document.getElementById("stopButton");
  const disableApprove = Boolean(payload.halted || payload.emergency_stop);
  approve.disabled = disableApprove || !(payload.reply_suggestions || []).length;
  sendDraft.disabled = Boolean(disableApprove || !payload.compose_text);
  reject.disabled = Boolean(payload.emergency_stop);
  stop.disabled = Boolean(payload.emergency_stop);
}

function renderState(payload, metricsPayload) {
  if (!payload?.classification) {
    return;
  }

  state.latestPayload = payload;
  state.emergencyStop = Boolean(payload.emergency_stop);

  const classification = payload.classification;
  document.getElementById("appName").textContent = classification.app;
  document.getElementById("screenName").textContent = classification.screen;
  document.getElementById("keyboardState").textContent = classification.keyboard_visible
    ? `Open (${classification.keyboard_height || 0}px)`
    : classification.keyboard_ambiguous
      ? "Ambiguous"
      : "Closed";
  document.getElementById("confidence").textContent = Number(classification.confidence).toFixed(2);
  document.getElementById("confidenceBar").style.width = `${Math.max(0, Math.min(100, classification.confidence * 100))}%`;
  document.getElementById("haltStatus").textContent = payload.halted ? "Halted" : "Ready";
  document.getElementById("summary").textContent = payload.summary || "No summary available.";
  document.getElementById("composePreview").textContent = payload.compose_text
    ? `Compose draft: ${payload.compose_text}`
    : "Compose draft is currently empty.";
  document.getElementById("lastAction").textContent = payload.last_action || "None";
  document.getElementById("lastActionResult").textContent = payload.last_action_result || "No execution yet.";
  document.getElementById("haltReason").textContent = payload.halt_reason || "";
  document.getElementById("screenshot").src = `/api/latest-screenshot?ts=${Date.now()}`;

  if (payload.halted) {
    showBanner(payload.halt_reason || "Unknown state halted the system.");
  } else {
    clearBanner();
  }

  const messages = document.getElementById("messages");
  messages.innerHTML = "";
  (classification.recent_messages || []).forEach((message) => {
    const item = document.createElement("li");
    item.className = "suggestion";
    item.textContent = message;
    messages.appendChild(item);
  });

  renderSuggestions(payload.reply_suggestions || []);
  renderReplyQuality(payload);
  renderVerificationErrors(payload.verification_errors || []);
  renderUnreadQueue(payload.unread_queue || state.automationState.queue_preview || []);
  renderThreadContext(payload.thread_context, payload);
  renderAutomationStatus(payload);
  renderMetrics(metricsPayload);
  updateButtons(payload);
}

async function refreshAll() {
  if (state.pollInFlight) {
    return;
  }

  state.pollInFlight = true;
  try {
    const [payload, metrics, automationState] = await Promise.all([
      api("/api/refresh-state", "POST"),
      api("/metrics").catch(() => ({})),
      api("/api/automation/state").catch(() => ({})),
    ]);
    state.automationState = automationState || {};
    renderState(payload, metrics);
  } catch (error) {
    showBanner(error.message);
    console.error("Refresh failed:", error);
  } finally {
    state.pollInFlight = false;
  }
}

async function runAction(path, body = null) {
  try {
    const payload = await api(path, "POST", body);
    if (payload?.classification) {
      const [metrics, automationState] = await Promise.all([
        api("/metrics").catch(() => ({})),
        api("/api/automation/state").catch(() => ({})),
      ]);
      state.automationState = automationState || {};
      renderState(payload, metrics);
      return;
    }

    if (payload?.error) {
      throw new Error(payload.error);
    }

    await refreshAll();
  } catch (error) {
    showBanner(error.message);
    console.error(`Action failed for ${path}:`, error);
  }
}

async function runStatusAction(path, body) {
  try {
    const payload = await api(path, "POST", body);
    if (payload?.error) {
      throw new Error(payload.error);
    }
    clearBanner(true);
    await refreshAll();
  } catch (error) {
    showBanner(error.message);
    console.error(`Status action failed for ${path}:`, error);
  }
}

async function setAutomationMode(mode, confidence) {
  await runAction("/api/automation/set-mode", { mode, confidence });
}

document.getElementById("refreshButton").onclick = () => refreshAll();
document.getElementById("approveTypeButton").onclick = () =>
  runAction("/api/approve/type-only", { suggestion_index: state.selectedSuggestionIndex });
document.getElementById("sendDraftButton").onclick = () => runAction("/api/approve/send-draft");
document.getElementById("rejectButton").onclick = () => runAction("/api/reject");
document.getElementById("stopButton").onclick = () => runAction("/api/emergency-stop");
document.querySelectorAll(".feedback-reason").forEach((button) => {
  button.addEventListener("click", () => {
    state.feedbackReason = button.dataset.reason || "other";
    document.querySelectorAll(".feedback-reason").forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
  });
});
document.getElementById("saveCorrectionButton").onclick = async () => {
  const payload = state.latestPayload || {};
  const selected = (payload.draft_candidates || [])[state.selectedSuggestionIndex] || {};
  const incoming = [...(payload.thread_context?.full_conversation || [])]
    .reverse()
    .find((entry) => entry.speaker === "other")?.text || (payload.classification?.recent_messages || []).at(-1) || "";
  const finalReply = document.getElementById("editedReplyInput").value.trim();
  if (!finalReply) {
    showBanner("Edited reply cannot be empty.");
    return;
  }
  await runStatusAction("/api/automation/feedback", {
    incoming,
    context: (payload.thread_context?.recent_messages || payload.classification?.recent_messages || []).slice(-6),
    relationship_type: payload.relationship_type || selected.relationship_type || "unknown",
    bad_ai_reply: selected.text || (payload.reply_suggestions || [])[state.selectedSuggestionIndex] || "",
    user_final_reply: finalReply,
    reason_bad: state.feedbackReason,
    selected_mode: (payload.automation_mode?.mode || "review").replace("-", "_"),
  });
};

refreshAll();
setInterval(refreshAll, 2500);
