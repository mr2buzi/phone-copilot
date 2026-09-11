const trainingState = {
  lastResponse: null,
  selectedIndex: 0,
  feedbackLabel: "approved",
  recentItems: [],
  styleReviewQueue: [],
  trainingLoopStatus: null,
};

const relationships = ["close_friend", "casual_friend", "romantic_interest", "family", "university", "professional", "unknown"];
const intents = ["auto", "greeting", "planning", "simple_question", "casual_checkin", "emotional", "joke_banter_safe", "argument", "romantic_flirty", "urgent", "professional", "apology", "thanks", "confirmation", "decline", "delay", "studying_work", "availability", "follow_up", "unknown"];
const diversityModes = ["conservative", "natural", "closest_style", "alternative_wording", "shorter", "more_direct"];
const parameters = [
  ["temperature", 0.7, 0, 1.2, 0.1],
  ["max_reply_length", 20, 3, 80, 1],
  ["slang_level", 0.6, 0, 1, 0.1],
  ["formality", 0.2, 0, 1, 0.1],
  ["directness", 0.7, 0, 1, 0.1],
  ["warmth", 0.5, 0, 1, 0.1],
  ["banter", 0.4, 0, 1, 0.1],
  ["question_bias", 0.5, 0, 1, 0.1],
  ["risk_tolerance", 0, 0, 1, 0.1],
];

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
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

function feedbackText(label) {
  return ({
    approved: "Marked as good. Save it to add this as approved training feedback.",
    too_formal: "Marked too formal. Edit the reply if needed, then save the training example.",
    too_long: "Marked too long. Shorten the correction, then save the training example.",
    not_me: "Marked not me. Write the reply you would actually send, then save.",
    missed_context: "Marked missed context. Add the missing detail in the correction or notes.",
    risky: "Marked too risky. This will stay review-only unless you save a safer correction.",
    too_flirty: "Marked too flirty. Save only a safe non-flirty correction.",
    too_dry: "Marked too dry. Add the tone you prefer, then save.",
    edited: "Ready for an edited correction. Change the reply, then save.",
    rejected: "Marked rejected. Save records negative feedback without adding a positive correction.",
  })[label] || "Feedback selected.";
}

function fillSelect(id, values, selected) {
  const select = document.getElementById(id);
  select.innerHTML = values.map((value) => `<option value="${value}" ${value === selected ? "selected" : ""}>${value}</option>`).join("");
}

function setupParameters() {
  const grid = document.getElementById("parameterGrid");
  grid.innerHTML = parameters
    .map(([name, value, min, max, step]) => `
      <div>
        <label>${name}</label>
        <input id="param_${name}" type="number" min="${min}" max="${max}" step="${step}" value="${value}" class="text-input" />
      </div>
    `)
    .join("") + `
      <div>
        <label>emoji_allowed</label>
        <input id="param_emoji_allowed" type="checkbox" />
      </div>
    `;
}

function collectRequest() {
  const params = {};
  parameters.forEach(([name]) => {
    params[name] = name === "max_reply_length"
      ? Number.parseInt(document.getElementById(`param_${name}`).value, 10)
      : Number(document.getElementById(`param_${name}`).value);
  });
  params.emoji_allowed = document.getElementById("param_emoji_allowed").checked;
  return {
    incoming: document.getElementById("incomingInput").value.trim(),
    context: document.getElementById("contextInput").value.split("\n").map((item) => item.trim()).filter(Boolean),
    contact_name: document.getElementById("contactInput").value.trim() || null,
    relationship_type: document.getElementById("relationshipSelect").value,
    intent_type: document.getElementById("intentSelect").value,
    diversity_mode: document.getElementById("diversitySelect").value,
    avoid_candidates: (trainingState.lastResponse?.candidates || []).map((candidate) => candidate.text),
    parameters: params,
  };
}

async function runChat(regenerate = false) {
  try {
    const request = collectRequest();
    if (!request.incoming) {
      showBanner("Incoming message is required.");
      return;
    }
    trainingState.lastResponse = await api(regenerate ? "/api/training/regenerate" : "/api/training/chat", "POST", request);
    trainingState.selectedIndex = 0;
    renderCandidates();
    await refreshStats();
    await refreshRecent();
  } catch (error) {
    showBanner(error.message);
  }
}

function renderCandidates() {
  const container = document.getElementById("candidateList");
  const response = trainingState.lastResponse;
  if (!response?.candidates?.length) {
    container.innerHTML = '<p class="summary">No candidates yet.</p>';
    return;
  }
  container.innerHTML = response.candidates.map((candidate, index) => {
    const scores = candidate.critic_scores || {};
    const retrieved = candidate.retrieved_examples || response.retrieved_examples || [];
    return `
      <div class="quality-candidate ${index === trainingState.selectedIndex ? "selected" : ""}">
        <strong>Candidate ${index + 1}</strong>
        <p>${escapeHtml(candidate.text)}</p>
        <small>style ${Number(scores.style_match_score || 0)} | relevance ${Number(scores.relevance_score || 0)} | risk ${Number(scores.risk_score || 0)} | ${escapeHtml(candidate.final_decision || "review")}</small>
        <small>${escapeHtml(candidate.blocked_reason || "No blocked reason")}</small>
        <small>${escapeHtml(response.retrieval_backend || "lexical")} | vector scores ${(response.retrieval_scores || []).join(", ")}</small>
        <details>
          <summary>Debug</summary>
          <pre>${escapeHtml(JSON.stringify({ retrieved, prompt_preview: response.prompt_preview, parameters: response.parameters }, null, 2))}</pre>
        </details>
        <div class="feedback-grid">
          ${feedbackButtons(index)}
        </div>
        <p class="status-info ${index === trainingState.selectedIndex ? "" : "hidden"}">${escapeHtml(index === trainingState.selectedIndex ? feedbackText(trainingState.feedbackLabel) : "")}</p>
      </div>
    `;
  }).join("");
  container.querySelectorAll("[data-feedback]").forEach((button) => {
    if (Number(button.dataset.index || 0) === trainingState.selectedIndex && button.dataset.feedback === trainingState.feedbackLabel) {
      button.classList.add("selected");
    }
    button.addEventListener("click", () => {
      trainingState.selectedIndex = Number(button.dataset.index || 0);
      trainingState.feedbackLabel = button.dataset.feedback;
      const candidate = trainingState.lastResponse.candidates[trainingState.selectedIndex];
      document.getElementById("correctReplyInput").value = candidate.text || "";
      document.getElementById("wrongReasonInput").value = button.dataset.feedback;
      showBanner(feedbackText(trainingState.feedbackLabel));
      renderCandidates();
    });
  });
}

function feedbackButtons(index) {
  return [
    ["approved", "Approve as good"],
    ["too_formal", "Too formal"],
    ["too_long", "Too long"],
    ["not_me", "Not me"],
    ["missed_context", "Missed context"],
    ["risky", "Too risky"],
    ["too_flirty", "Too flirty"],
    ["too_dry", "Too dry"],
    ["edited", "Good but edit"],
    ["rejected", "Reject"],
  ].map(([value, label]) => `<button type="button" class="secondary compact-button" data-index="${index}" data-feedback="${value}">${label}</button>`).join("");
}

async function saveFeedback() {
  try {
    const response = trainingState.lastResponse;
    const candidate = response?.candidates?.[trainingState.selectedIndex];
    if (!response || !candidate) {
      showBanner("Generate a candidate first.");
      return;
    }
    const request = collectRequest();
    const saveToCorrections = document.getElementById("saveCorrectionsCheckbox").checked && trainingState.feedbackLabel !== "rejected";
    const payload = {
      incoming: request.incoming,
      context: request.context,
      contact_name: request.contact_name,
      relationship_type: response.relationship_type || request.relationship_type,
      intent_type: response.intent_type || request.intent_type,
      ai_reply: candidate.text,
      correct_reply: document.getElementById("correctReplyInput").value.trim(),
      feedback_label: trainingState.feedbackLabel,
      notes: document.getElementById("notesInput").value.trim(),
      parameters: request.parameters,
      retrieved_example_ids: (candidate.retrieval_ids_used || []),
      save_to_corrections: saveToCorrections,
      add_to_vector_db: document.getElementById("addVectorCheckbox").checked && saveToCorrections,
    };
    const result = await api("/api/training/feedback", "POST", payload);
    showBanner(result.vector_rebuild_required ? "Saved. Rebuild vector DB required." : "Saved training feedback.");
    await refreshStats();
    await refreshRecent();
  } catch (error) {
    showBanner(error.message);
  }
}

async function refreshStats() {
  const stats = await api("/api/training/stats").catch(() => ({}));
  document.getElementById("trainingStats").innerHTML = `
    <div><span>Corrections</span><strong>${Number(stats.corrections_count || 0)}</strong></div>
    <div><span>Real examples</span><strong>${Number(stats.real_examples_count || 0)}</strong></div>
    <div><span>Synthetic</span><strong>${Number(stats.synthetic_examples_count || 0)}</strong></div>
    <div><span>Vector rows</span><strong>${Number(stats.vector_index_count || 0)}</strong></div>
    <div><span>Rebuild</span><strong>${stats.vector_rebuild_required ? "Required" : "Clean"}</strong></div>
    <div><span>Last rebuild</span><strong>${escapeHtml(stats.last_vector_rebuild || "Unknown")}</strong></div>
  `;
  await refreshTrainingControlStatus();
}

async function refreshTrainingControlStatus() {
  const status = await api("/api/training/training-status").catch(() => ({}));
  const latest = status.last_eval || {};
  const previous = status.previous_eval || {};
  const latestScore = Number(latest.average_overall_score || 0);
  const previousScore = Number(previous.avg_overall_score || 0);
  const delta = previousScore ? (latestScore - previousScore).toFixed(2) : "0.00";
  const container = document.getElementById("trainingControlStats");
  if (!container) return;
  container.innerHTML = `
    <div><span>Last score</span><strong>${latestScore || "Unknown"}</strong></div>
    <div><span>Delta</span><strong>${escapeHtml(delta)}</strong></div>
    <div><span>Pass / fail</span><strong>${Number(latest.pass_count || 0)} / ${Number(latest.fail_count || 0)}</strong></div>
    <div><span>Review queue</span><strong>${Number(status.review_queue_count || 0)}</strong></div>
    <div><span>Vector rebuild</span><strong>${status.vector_rebuild_required ? "Required" : "Clean"}</strong></div>
    <div><span>Corrections</span><strong>${Number(status.approved_corrections_count || 0)}</strong></div>
  `;
  await refreshTrainingLoopStatus();
}

async function refreshTrainingLoopStatus() {
  const status = await api("/api/training/loop/status").catch(() => ({}));
  trainingState.trainingLoopStatus = status;
  const config = status.config || {};
  const lastResult = status.last_result || {};
  const container = document.getElementById("trainingLoopStats");
  if (!container) return;
  container.innerHTML = `
    <div><span>Loop</span><strong>${status.enabled ? "Enabled" : "Disabled"}</strong></div>
    <div><span>State</span><strong>${status.running ? "Running" : "Idle"}</strong></div>
    <div><span>Last run</span><strong>${escapeHtml(status.last_finished_at || "Never")}</strong></div>
    <div><span>Next run</span><strong>${escapeHtml(status.next_run_at || "Not scheduled")}</strong></div>
    <div><span>Review queue</span><strong>${Number(status.review_queue_count || 0)}</strong></div>
    <div><span>Active learning</span><strong>${Number(status.active_learning_queue_count || 0)}</strong></div>
    <div><span>Vector rebuild</span><strong>${status.vector_rebuild_required ? "Required" : "Clean"}</strong></div>
    <div><span>Last result</span><strong>${Number(lastResult.review_candidates_generated || 0)} candidates</strong></div>
  `;
  document.getElementById("loopIntervalSelect").value = String(config.interval_minutes || 30);
  document.getElementById("loopMaxCasesInput").value = Number(config.max_cases_per_run || 100);
  document.getElementById("loopMaxCandidatesInput").value = Number(config.max_review_candidates_per_run || 25);
  document.getElementById("loopAutoRebuildCheckbox").checked = config.auto_rebuild_after_approval !== false;
  document.getElementById("loopQuietHoursCheckbox").checked = Boolean(config.quiet_hours?.enabled);
  const log = document.getElementById("trainingLoopLog");
  if (log && status.last_error) {
    log.textContent = `Last error: ${status.last_error}`;
  }
}

function collectTrainingLoopConfig() {
  return {
    interval_minutes: Number.parseInt(document.getElementById("loopIntervalSelect").value, 10),
    max_cases_per_run: Number.parseInt(document.getElementById("loopMaxCasesInput").value, 10),
    max_review_candidates_per_run: Number.parseInt(document.getElementById("loopMaxCandidatesInput").value, 10),
    auto_rebuild_after_approval: document.getElementById("loopAutoRebuildCheckbox").checked,
    quiet_hours: {
      enabled: document.getElementById("loopQuietHoursCheckbox").checked,
      start: "23:00",
      end: "08:00",
    },
    auto_approve: false,
  };
}

async function trainingLoopAction(path, label, body = null) {
  const log = document.getElementById("trainingLoopLog");
  try {
    log.textContent = `${label}...`;
    const result = await api(path, "POST", body);
    log.textContent = JSON.stringify(result, null, 2);
    await refreshStats();
    await refreshStyleReviewQueue();
  } catch (error) {
    log.textContent = error.message;
    showBanner(error.message);
  }
}

async function runTrainingControl(path, label) {
  const log = document.getElementById("trainingControlLog");
  try {
    log.textContent = `${label} running...`;
    const result = await api(path, "POST");
    log.textContent = JSON.stringify({
      status: result.status,
      exit_code: result.exit_code,
      stdout: result.stdout,
      stderr: result.stderr,
      report: result.report,
    }, null, 2);
    await refreshStats();
    await refreshStyleReviewQueue();
  } catch (error) {
    log.textContent = error.message;
    showBanner(error.message);
  }
}

async function refreshRecent() {
  const payload = await api("/api/training/recent?limit=50").catch(() => ({ items: [] }));
  trainingState.recentItems = payload.items || [];
  const container = document.getElementById("recentFeedback");
  container.innerHTML = trainingState.recentItems.length ? trainingState.recentItems.slice().reverse().map((item) => `
    <div class="debug-example">
      <small>${escapeHtml(item.timestamp || "")} | ${escapeHtml(item.relationship_type || "")} | ${escapeHtml(item.intent_type || "")} | ${escapeHtml(item.feedback_label || "")}</small>
      <p>${escapeHtml(item.incoming || "")}</p>
      <p>${escapeHtml(item.selected_candidate || "")} -> ${escapeHtml(item.correct_reply || "")}</p>
      <button type="button" class="secondary compact-button" data-quarantine="${escapeHtml(item.id || "")}">Quarantine</button>
    </div>
  `).join("") : '<p class="summary">No recent training feedback.</p>';
  container.querySelectorAll("[data-quarantine]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.quarantine;
      if (!id) return;
      await api("/api/training/quarantine-example", "POST", { example_id: id, reason: "quarantined from training page" });
      showBanner("Training example quarantined. Rebuild vector DB required.");
      await refreshStats();
      await refreshRecent();
    });
  });
}

async function refreshStyleReviewQueue() {
  const payload = await api("/api/training/style-review-queue").catch(() => ({ items: [] }));
  trainingState.styleReviewQueue = payload.items || [];
  const container = document.getElementById("styleReviewQueue");
  container.innerHTML = trainingState.styleReviewQueue.length ? trainingState.styleReviewQueue.map((item) => `
    <div class="debug-example">
      <small>${escapeHtml(item.timestamp || "")} | ${escapeHtml(item.relationship_type || "")} | ${escapeHtml(item.intent_type || "")} | ${escapeHtml(item.status || "needs_human_review")}</small>
      <p><strong>Incoming:</strong> ${escapeHtml(item.incoming || "")}</p>
      <p><strong>Bad AI:</strong> ${escapeHtml(item.bad_ai_reply || "")}</p>
      <label>Suggested reply</label>
      <textarea class="text-input correction-textarea" data-style-edit="${escapeHtml(item.row_id || "")}">${escapeHtml(item.suggested_better_reply || "")}</textarea>
      <p class="summary">Style score ${Number(item.style_score || 0)} | ${escapeHtml((item.reason_bad || "unknown").toString())}</p>
      <p class="summary">${escapeHtml((item.improvement_notes || []).join(" • ") || "No notes")}</p>
      <div class="button-row">
        <button type="button" class="primary compact-button" data-style-approve="${escapeHtml(item.row_id || "")}">Approve</button>
        <button type="button" class="secondary compact-button" data-style-reject="${escapeHtml(item.row_id || "")}">Reject</button>
      </div>
    </div>
  `).join("") : '<p class="summary">No style review items pending.</p>';

  container.querySelectorAll("[data-style-approve]").forEach((button) => {
    button.addEventListener("click", async () => {
      const rowId = button.dataset.styleApprove;
      const textarea = container.querySelector(`[data-style-edit="${CSS.escape(rowId || "")}"]`);
      await api("/api/training/style-review-queue/approve", "POST", {
        row_id: rowId,
        edited_reply: textarea?.value?.trim() || null,
      });
      showBanner("Style review approved into corrections.");
      await refreshStyleReviewQueue();
      await refreshStats();
    });
  });

  container.querySelectorAll("[data-style-reject]").forEach((button) => {
    button.addEventListener("click", async () => {
      const rowId = button.dataset.styleReject;
      const reason = window.prompt("Reject reason", "not good enough") || "rejected";
      await api("/api/training/style-review-queue/reject", "POST", {
        row_id: rowId,
        reason,
      });
      showBanner("Style review rejected.");
      await refreshStyleReviewQueue();
      await refreshStats();
    });
  });
}

async function rebuildVector() {
  try {
    showBanner("Rebuilding vector DB...");
    const result = await api("/api/training/rebuild-vector-db", "POST");
    showBanner(`Vector DB rebuilt: ${result.rows_indexed || 0} rows indexed.`);
    await refreshStats();
  } catch (error) {
    showBanner(error.message);
  }
}

fillSelect("relationshipSelect", relationships, "unknown");
fillSelect("intentSelect", intents, "auto");
fillSelect("diversitySelect", diversityModes, "natural");
setupParameters();
document.getElementById("chatButton").onclick = () => runChat(false);
document.getElementById("regenerateButton").onclick = () => runChat(true);
document.getElementById("saveFeedbackButton").onclick = saveFeedback;
document.getElementById("rebuildVectorButton").onclick = rebuildVector;
document.getElementById("refreshReviewQueueButton").onclick = refreshStyleReviewQueue;
document.getElementById("runEvaluationButton").onclick = () => runTrainingControl("/api/training/run-evaluation?limit=100", "Evaluation");
document.getElementById("generateImprovementsButton").onclick = () => runTrainingControl("/api/training/generate-improvements?limit=100", "Improvement generation");
document.getElementById("controlRebuildVectorButton").onclick = rebuildVector;
document.getElementById("runIterationButton").onclick = () => runTrainingControl("/api/training/run-iteration?limit=100", "Training iteration");
document.getElementById("startTrainingLoopButton").onclick = () => trainingLoopAction("/api/training/loop/start", "Starting autonomous loop");
document.getElementById("stopTrainingLoopButton").onclick = () => trainingLoopAction("/api/training/loop/stop", "Stopping autonomous loop");
document.getElementById("runTrainingLoopNowButton").onclick = () => trainingLoopAction("/api/training/loop/run-now", "Starting one loop run");
document.getElementById("saveTrainingLoopConfigButton").onclick = () => trainingLoopAction("/api/training/loop/config", "Saving loop config", collectTrainingLoopConfig());
refreshStats();
refreshRecent();
refreshStyleReviewQueue();
