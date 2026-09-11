const simpleState = {
  lastResponse: null,
  selectedIndex: 0,
  feedbackLabel: "approved",
};

const relationships = ["close_friend", "casual_friend", "family", "university", "professional", "unknown", "romantic_interest"];
const intents = ["auto", "greeting", "planning", "simple_question", "casual_checkin", "apology", "thanks", "confirmation", "decline", "delay", "studying_work", "availability", "follow_up", "unknown"];

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

function fillSelect(id, values, selected) {
  const select = document.getElementById(id);
  select.innerHTML = values.map((value) => `<option value="${value}" ${value === selected ? "selected" : ""}>${value}</option>`).join("");
}

function feedbackText(label) {
  return ({
    approved: "Marked good. Save to keep this as approved feedback.",
    too_long: "Marked too long. Shorten the correction, then save.",
    not_me: "Marked not me. Write your version, then save.",
    missed_context: "Marked missed context. Add the missing context in notes.",
    risky: "Marked risky. Save a safer correction or reject it.",
    rejected: "Marked rejected. Saving will record feedback without adding a positive correction.",
  })[label] || "Feedback selected.";
}

function collectRequest() {
  return {
    incoming: document.getElementById("incomingInput").value.trim(),
    context: [],
    contact_name: null,
    relationship_type: document.getElementById("relationshipSelect").value,
    intent_type: document.getElementById("intentSelect").value,
    diversity_mode: "natural",
    avoid_candidates: (simpleState.lastResponse?.candidates || []).map((candidate) => candidate.text),
    parameters: {
      temperature: 0.7,
      max_reply_length: 20,
      slang_level: 0.6,
      formality: 0.2,
      directness: 0.7,
      warmth: 0.5,
      banter: 0.4,
      emoji_allowed: false,
      question_bias: 0.5,
      risk_tolerance: 0.0,
    },
  };
}

async function runChat(regenerate = false) {
  try {
    const request = collectRequest();
    if (!request.incoming) {
      showBanner("Incoming message is required.");
      return;
    }
    simpleState.lastResponse = await api(regenerate ? "/api/training/regenerate" : "/api/training/chat", "POST", request);
    simpleState.selectedIndex = 0;
    simpleState.feedbackLabel = "approved";
    renderCandidates();
    await refreshStats();
  } catch (error) {
    showBanner(error.message);
  }
}

function renderCandidates() {
  const container = document.getElementById("candidateList");
  const response = simpleState.lastResponse;
  if (!response?.candidates?.length) {
    container.innerHTML = '<p class="summary">No candidates yet.</p>';
    return;
  }
  container.innerHTML = response.candidates.map((candidate, index) => `
    <div class="quality-candidate ${index === simpleState.selectedIndex ? "selected" : ""}">
      <strong>Candidate ${index + 1}</strong>
      <p>${escapeHtml(candidate.text)}</p>
      <small>${escapeHtml(candidate.final_decision || "review")} | ${escapeHtml(candidate.blocked_reason || "No blocked reason")}</small>
      <div class="feedback-grid">
        ${feedbackButtons(index)}
      </div>
      <p class="status-info ${index === simpleState.selectedIndex ? "" : "hidden"}">${escapeHtml(index === simpleState.selectedIndex ? feedbackText(simpleState.feedbackLabel) : "")}</p>
    </div>
  `).join("");
  container.querySelectorAll("[data-feedback]").forEach((button) => {
    if (Number(button.dataset.index || 0) === simpleState.selectedIndex && button.dataset.feedback === simpleState.feedbackLabel) {
      button.classList.add("selected");
    }
    button.addEventListener("click", () => {
      simpleState.selectedIndex = Number(button.dataset.index || 0);
      simpleState.feedbackLabel = button.dataset.feedback;
      const candidate = simpleState.lastResponse.candidates[simpleState.selectedIndex];
      document.getElementById("correctReplyInput").value = candidate.text || "";
      showBanner(feedbackText(simpleState.feedbackLabel));
      renderCandidates();
    });
  });
}

function feedbackButtons(index) {
  return [
    ["approved", "Good"],
    ["too_long", "Too long"],
    ["not_me", "Not me"],
    ["missed_context", "Missed context"],
    ["risky", "Risky"],
    ["rejected", "Reject"],
  ].map(([value, label]) => `<button type="button" class="secondary compact-button" data-index="${index}" data-feedback="${value}">${label}</button>`).join("");
}

async function saveFeedback() {
  try {
    const response = simpleState.lastResponse;
    const candidate = response?.candidates?.[simpleState.selectedIndex];
    if (!response || !candidate) {
      showBanner("Generate a candidate first.");
      return;
    }
    const request = collectRequest();
    const saveToCorrections = document.getElementById("saveCorrectionsCheckbox").checked && simpleState.feedbackLabel !== "rejected";
    const result = await api("/api/training/feedback", "POST", {
      incoming: request.incoming,
      context: [],
      contact_name: null,
      relationship_type: response.relationship_type || request.relationship_type,
      intent_type: response.intent_type || request.intent_type,
      ai_reply: candidate.text,
      correct_reply: document.getElementById("correctReplyInput").value.trim(),
      feedback_label: simpleState.feedbackLabel,
      notes: document.getElementById("notesInput").value.trim(),
      parameters: request.parameters,
      retrieved_example_ids: candidate.retrieval_ids_used || [],
      save_to_corrections: saveToCorrections,
      add_to_vector_db: document.getElementById("addVectorCheckbox").checked && saveToCorrections,
    });
    showBanner(result.vector_rebuild_required ? "Saved. Rebuild vector DB required." : "Saved feedback.");
    await refreshStats();
  } catch (error) {
    showBanner(error.message);
  }
}

async function refreshStats() {
  const stats = await api("/api/training/stats").catch(() => ({}));
  document.getElementById("trainingStats").innerHTML = `
    <div><span>Corrections</span><strong>${Number(stats.corrections_count || 0)}</strong></div>
    <div><span>Synthetic</span><strong>${Number(stats.synthetic_examples_count || 0)}</strong></div>
    <div><span>Vector rows</span><strong>${Number(stats.vector_index_count || 0)}</strong></div>
    <div><span>Rebuild</span><strong>${stats.vector_rebuild_required ? "Required" : "Clean"}</strong></div>
  `;
}

fillSelect("relationshipSelect", relationships, "unknown");
fillSelect("intentSelect", intents, "auto");
document.getElementById("chatButton").onclick = () => runChat(false);
document.getElementById("regenerateButton").onclick = () => runChat(true);
document.getElementById("saveFeedbackButton").onclick = saveFeedback;
refreshStats();
