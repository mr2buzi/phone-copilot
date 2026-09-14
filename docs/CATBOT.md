# Catbot conversation recovery

The full controller serves Catbot at `/catbot`. Its chat endpoint is
`POST /api/training/catbot/chat`. See [SETUP.md](SETUP.md) to run the controller.
Restart an existing controller after updating the code.

## Casual follow-ups

Catbot now recognizes these messages in the romantic-interest simulator path:

| Input | Response behavior |
| --- | --- |
| `im good wys`, `wys?`, `what u saying`, `what you saying` | Answer the status/activity question with a concrete detail. |
| `bruh`, including trailing `icl` | Treat a short reaction after a bot reply as a repair callout when there is no recognized story context. |
| `fix up`, `fix up then` | Acknowledge the callout and select a repair response that has not already appeared in the recent conversation. |

Standalone `bruh` without prior conversation is not automatically classified as a
complaint. Story reactions retain their existing classification. Phrases such as
`i need to fix up my bike` do not trigger the short-callout rule. Matching ignores
case and surrounding sentence punctuation; `wys` is matched as a complete word.

## Why the 502 happened

The reported sequence was `hi` -> `hey u alright` -> `im good wys`, followed by
`bruh` and `fix up`. All three follow-ups previously reached the unclassified
reply plan. That plan required a specific continuation but did not know which
question or complaint to answer. Provider replies could repeatedly fail with
`generic_ai_style` or `not_carrying_conversation`, eventually exhausting repairs
and returning HTTP 502.

The fix routes the status slang through the existing reciprocal-question plan
and the short complaints through the quality-complaint plan. It preserves the
complaint subtype during slot extraction and supplies several local repair
options so consecutive callouts do not reuse one apology. These recognized plans
can answer locally and can also repair rejected or unavailable provider output.

Every selected reply still passes the existing content, reply-plan and repetition
checks. Candidates remain review-only, with `auto_send_allowed: false`; the
Catbot chat route does not touch ADB. Local planned replies report
`provider: "plan_ranker"` and `external_api_used: false`.

This is a targeted repair, not a guarantee that every possible message has a
local answer. Open-ended generation still requires a configured provider. HTTP
502 remains appropriate when all provider attempts and available local repairs
fail validation. The existing no-usable-reply regression continues to enforce
that behavior.

## Verification

Run from the repository root in the project's Python environment:

```sh
python -m pytest -q tests/test_training_page.py -k "casual_followup or bare_reaction"
python -m pytest -q
```

The focused tests cover spelling/case variants, story and ordinary-sentence
exclusions, and the three-message follow-up sequence under generic replies,
empty provider failures and replies that are too short. They verify that replies
are distinct, pass validation and remain review-only. Provider responses are
stubbed, so these tests need no credentials or external network access.

For a manual check, start a new Catbot conversation and send `hi`, `im good wys`,
`bruh`, and `fix up`, waiting for each reply. The status question should get an
activity answer and both callouts should get distinct repair replies. If another
message fails, inspect `window.catbotLastRoutePayload` in the browser console and
record the incoming text, HTTP status and rejection reason. Keep private
conversation content and credentials out of public bug reports.
