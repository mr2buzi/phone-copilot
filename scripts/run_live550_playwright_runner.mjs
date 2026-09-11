import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const resultPath = "tmp_live550_playwright_result.json";

const seed = [
  "hi", "how are u", "im good u", "ive been chilling wby baby", "u already asked silly",
  "u already asked how my day was / how i am", "hi", "hru", "what u up to", "what did u doo",
  "omg im so bored", "why u asking stupid qs for", "forget that anyway", "im horny", "come here then",
  "need ur lips on my neck", "what else", "mhmmm", "i missed you", "really", "how much",
  "i need you", "i want u", "tell me then", "go on then", "and?", "prove it", "how so",
  "hey", "hey you", "wyd", "what u been up to", "im tired icl", "why u tired",
  "how was ur day", "what did u do today", "what did u do in gym", "what u training", "i hate legs",
  "same tbh", "do u box", "is boxing hard", "would u fight me", "behave", "lol why",
  "where u been", "wyd later", "u doing anything nice", "trust me", "u good", "are u okay",
  "why u being weird", "i asked u a question", "are u gonna answer", "nth wby", "i js asked u a question",
  "lol im chilling", "yeah why", "what have u been doing", "i been busy asf", "what dyu study",
  "what do u study", "what uni", "what course", "what do u do", "what project u working on",
  "where u from", "how old r u", "what car u like", "what about u", "whats ur dream",
  "do u pray", "guess what", "tell me then", "i was at home yeah and showering and some guy came in and started running in my living room",
  "bruh wtf", "nah like actually", "what would u do", "im bored", "entertain me", "u tell me",
  "u pick", "talk then", "idk talk", "u already asked that", "ur boring me", "ur repeating urself",
  "why u keep asking that", "answer properly", "stop asking questions", "thats dry mate", "be more specific",
  "same", "nice", "oh fairs", "wdym", "what do u mean", "why", "why though", "fr?",
  "no like actually", "okay then", "go on", "carry on", "continue", "say it then", "and then",
  "what else would u do", "mhmm keep going", "thats it?", "that all?", "nah tell me properly",
  "im in bed", "im in bed rn", "just chilling wbu", "im good wbu", "im alright wby",
  "been busy today", "just got back from gym", "just woke up", "about to sleep", "thinking about u",
  "u miss me?", "do u want me?", "how bad", "show me then", "come closer", "kiss me then",
];

const target = 550;
const variants = ["", " icl", " lol", " ngl", " tbh", " lowk", " btw"];
const messages = Array.from({ length: target }, (_, index) => {
  let message = seed[index % seed.length];
  if (index >= seed.length && index % 19 === 0) message += variants[index % variants.length];
  return message.trim();
});

const normalize = (value) => String(value || "").replace(/\s+/g, " ").trim().toLowerCase();
const includesAny = (text, terms) => terms.some((term) => text.includes(term));
const candidate = (payload) => payload?.candidate || payload?.candidates?.[0] || {};
const replyParts = (payload) => (
  Array.isArray(payload?.reply_sequence) && payload.reply_sequence.length
    ? payload.reply_sequence
    : Array.isArray(payload?.candidate?.sequence) && payload.candidate.sequence.length
      ? payload.candidate.sequence
      : [payload?.reply || ""]
);
const candidatePath = (item) => (
  item.catbot_ai_repair_accepted || item.catbot_ai_repaired
    ? "repair"
    : item.catbot_ai_retry_accepted || item.catbot_ai_initial_reject_reason
      ? "retry"
      : "first"
);

function failReason(user, bot, payload, routePayload) {
  const item = candidate(payload);
  const reply = normalize(bot);
  const userNorm = normalize(user);
  const parts = replyParts(payload).map((part) => String(part || "").trim()).filter(Boolean);
  if (!routePayload || routePayload.ok === false) return `route_error:${routePayload?.detail || routePayload?.error || routePayload?.status || "unknown"}`;
  if (!payload) return "missing_route_payload";
  if (!reply) return "empty_visible_reply";
  if (/catbot error/i.test(bot)) return "ui_error_reply";
  if (item.fallback_used || item.manual_review_fallback) return "fallback_used";
  if (item.provider === "deterministic" || item.model === "fallback") return "deterministic_fallback_provider";
  if (item.catbot_ai_reject_reason) return `leaked_reject:${item.catbot_ai_reject_reason}`;
  if (item.reply_plan_validation) return `plan_validation:${item.reply_plan_validation}`;
  if (/^(ok|okay|k|yeah|yh|lol|haha|fair|nice)$/.test(reply)) return "dead_one_word_reply";
  if (includesAny(reply, ["what do you want to talk about", "what should we talk about", "give me a topic"])) return "asks_user_to_carry_conversation";
  if (includesAny(userNorm, ["u already asked", "ur repeating", "keep asking", "stop asking"])) {
    const asksAgain = reply.endsWith("?") || includesAny(reply, ["what's", "what u", "how was", "how are", "wyd", "wby", "wbu"]);
    const namesQuestionWithoutAsking = includesAny(reply, ["no more", "not asking", "stop asking", "questions from me"]);
    if (asksAgain && !namesQuestionWithoutAsking) return "loop_callout_asked_another_question";
  }
  if (/mh+m+|what else|and then|keep going|that all|that it/.test(userNorm) && includesAny(reply, ["what else u want", "what do u want", "tell me what"])) return "adult_followup_meta_question";
  if (/^(im horny|i need you|i want u)$/.test(userNorm) && parts.length < 3) return "direct_adult_too_few_bubbles";
  if (/(im good wbu|im good u|just chilling wbu|ive been chilling wby)/.test(userNorm)
    && /^(?:im good|im fine|just chilling|same just chilling)(?:\s*\/\s*|\s+)(?:u|you|wby|wbu|hru|hbu)\??$/.test(reply)) {
    return "bare_status_mirror";
  }
  return "";
}

function writeResult(result) {
  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2));
}

async function readRoutePayload(page, incoming) {
  for (let tick = 0; tick < 240; tick += 1) {
    await page.waitForTimeout(250);
    let wrapped = {};
    try {
      wrapped = await page.evaluate(() => window.catbotLastRoutePayload || {});
      if (!wrapped || typeof wrapped !== "object") {
        const raw = document.getElementById("catbotLastRoutePayload")?.textContent || "{}";
        wrapped = JSON.parse(raw || "{}");
      }
    } catch {
      try {
        const raw = await page.locator("#catbotLastRoutePayload").textContent({ timeout: 5000 });
        wrapped = JSON.parse(raw || "{}");
      } catch {
        wrapped = { ok: false, detail: "invalid_route_payload_metadata" };
      }
    }
    if (wrapped.incoming === incoming && !wrapped.pending) return wrapped;
  }
  return { ok: false, detail: "route_payload_metadata_timeout", incoming };
}

async function main() {
  fs.rmSync(resultPath, { force: true });
  const browser = await chromium.launch({
    headless: false,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto("http://127.0.0.1:8765/catbot?live550&debounce_ms=250", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Reset thread" }).click();
  const metrics = {
    total_turns: 0,
    accepted_first_try: 0,
    accepted_after_retry: 0,
    accepted_after_repair: 0,
    failures: 0,
  };
  const rows = [];
  for (let index = 0; index < target; index += 1) {
    const user = messages[index];
    const before = await page.locator(".catbot-message-row .catbot-message").allTextContents();
    await page.getByPlaceholder("message").fill(user);
    await page.getByRole("button", { name: "Send" }).click();
    const routePayload = await readRoutePayload(page, user);
    const payload = routePayload.response || routePayload;
    const after = await page.locator(".catbot-message-row .catbot-message").allTextContents();
    const newRows = after.slice(before.length);
    const bot = newRows.slice(1).join(" / ").trim();
    const item = candidate(payload);
    const path = candidatePath(item);
    const reason = failReason(user, bot, payload, routePayload);
    const row = {
      turn: index + 1,
      user,
      bot,
      move: item.reply_plan_move || "unknown",
      shape: item.reply_plan_shape || "",
      path,
      reason,
      reject: item.catbot_ai_initial_reject_reason || "",
      finalRepairReason: item.catbot_ai_final_repair_reason || "",
      provider: item.provider || "",
      timing_ms: item.timing_ms || null,
      provider_latency_ms: item.provider_latency_ms || null,
    };
    rows.push(row);
    if (reason) {
      metrics.failures += 1;
      const result = { index, target, failed: true, done: false, failure: row, metrics, recent: rows.slice(-12) };
      writeResult(result);
      console.log(JSON.stringify(result, null, 2));
      await browser.close();
      process.exit(1);
    }
    metrics.total_turns += 1;
    if (path === "repair") metrics.accepted_after_repair += 1;
    else if (path === "retry") metrics.accepted_after_retry += 1;
    else metrics.accepted_first_try += 1;
    if ((index + 1) % 10 === 0) {
      const result = { index: index + 1, target, failed: false, done: false, failure: null, metrics, recent: rows.slice(-5) };
      writeResult(result);
      console.log(`progress ${index + 1}/${target}`);
    }
  }
  const result = { index: target, target, failed: false, done: true, failure: null, metrics, recent: rows.slice(-12) };
  writeResult(result);
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
}

main().catch((error) => {
  const result = { failed: true, done: false, error: String(error && error.stack || error) };
  writeResult(result);
  console.error(JSON.stringify(result, null, 2));
  process.exit(1);
});
