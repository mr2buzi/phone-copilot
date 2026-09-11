const $ = (id) => document.getElementById(id);
let scenarios = [],
  current = null,
  busy = false;

async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Request failed");
  return result;
}
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function setBusy(value) {
  busy = value;
  ["generate", "reset", "scenario"].forEach((id) => {
    $(id).disabled = value;
  });
  document.querySelectorAll(".candidate button").forEach((button) => {
    button.disabled = value;
  });
}
function selectScenario() {
  current = scenarios.find((item) => item.id === $("scenario").value);
  $("contact").textContent = current.contact;
  $("relationship").textContent = current.relationship.replaceAll("_", " ");
  $("messages").replaceChildren();
  for (const message of current.messages) {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message;
    $("messages").append(bubble);
  }
  $("candidates").replaceChildren();
  $("approval").hidden = true;
  $("scores").textContent = "No draft selected";
  $("elapsed").textContent = "Idle";
  $("detail").open = false;
  document
    .querySelectorAll("#stages li")
    .forEach((item) => item.classList.remove("complete"));
  document.querySelectorAll("#stages small")[2].textContent = "Awaiting input";
  document.querySelectorAll("#stages small")[3].textContent = "Review required";
  status("Ready");
}
async function generate() {
  if (busy) return;
  setBusy(true);
  status("Generating drafts...");
  $("approval").hidden = true;
  try {
    const result = await api("/api/draft", { scenario: current.id });
    $("candidates").replaceChildren();
    result.bundle.reply_candidates.slice(0, 3).forEach((candidate, index) => {
      const item = document.createElement("article");
      item.className = "candidate";
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = `0${index + 1}`;
      const content = document.createElement("div");
      const text = document.createElement("p");
      text.textContent = candidate.text;
      const meta = document.createElement("div");
      meta.className = "meta";
      const score = candidate.score_breakdown?.final_confidence;
      meta.textContent = `${typeof score === "number" ? "Rank score " + score.toFixed(2) : "Ranked candidate"} / Review required`;
      content.append(text, meta);
      const approve = document.createElement("button");
      approve.textContent = "Approve";
      approve.setAttribute("aria-label", `Approve candidate ${index + 1}`);
      approve.addEventListener("click", async () => {
        if (busy) return;
        setBusy(true);
        try {
          const approved = await api("/api/approve", {
            scenario: current.id,
            candidate_index: index,
          });
          $("approved-text").textContent = approved.text;
          $("approval").hidden = false;
          $("scores").textContent = JSON.stringify(
            candidate.score_breakdown,
            null,
            2,
          );
          item.classList.add("selected");
          document.querySelectorAll("#stages li")[3].classList.add("complete");
          document.querySelectorAll("#stages small")[3].textContent =
            "Approved locally";
          status("Approved locally. No message sent.");
          setBusy(false);
          document.querySelectorAll(".candidate button").forEach((button) => {
            button.disabled = true;
          });
        } catch (error) {
          status(error.message, true);
          setBusy(false);
        }
      });
      item.append(rank, content, approve);
      $("candidates").append(item);
    });
    $("scores").textContent = JSON.stringify(
      result.bundle.reply_candidates[0]?.score_breakdown || {},
      null,
      2,
    );
    $("elapsed").textContent = `${result.elapsed_ms} ms`;
    document
      .querySelectorAll("#stages li")
      .forEach((item, index) => item.classList.toggle("complete", index < 3));
    document.querySelectorAll("#stages small")[2].textContent =
      `${result.bundle.reply_candidates.length} ranked candidates`;
    document.querySelectorAll("#stages small")[3].textContent =
      "Review required";
    status(`${result.bundle.reply_candidates.length} drafts ready for review`);
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
}
$("generate").addEventListener("click", generate);
$("scenario").addEventListener("change", selectScenario);
$("reset").addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try {
    await api("/api/reset", {});
    selectScenario();
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
});
api("/api/scenarios")
  .then((result) => {
    scenarios = result.scenarios;
    for (const item of scenarios) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.title;
      $("scenario").append(option);
    }
    selectScenario();
    setBusy(false);
  })
  .catch((error) => status(error.message, true));
