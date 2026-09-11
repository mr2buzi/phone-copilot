# Architecture

Phone Copilot has two entry points: the full controller for local device work and a recording demo with no device executor. Both use the same drafting library.

## Observe, plan, review, execute

`apps/controller/main.py` exposes FastAPI routes and the static desktop views. `PhoneCopilotService` coordinates ADB observation, perception, drafting, the state-machine planner, policy checks and SQLite logging.

`libs/adb/` wraps device commands. `libs/perception/` combines screenshots, OCR and the Android UI hierarchy into screen classifications. The planner produces typed execution steps with expected screen state and time limits. Policy evaluation and explicit approval sit before device execution. The executor checks preconditions and observes the resulting screen after acting.

Screen confidence and reply confidence answer different questions. The first describes the observed UI; the second is a heuristic score for a text candidate. Neither proves that sending is appropriate.

## Drafting and retrieval

The drafting service determines relationship and intent, builds conversation state, retrieves relevant examples and produces candidate replies. A model adapter can generate text when enabled. A deterministic path supports local fallback behavior and the recording demo.

Lexical retrieval is the dependency-light default. Chroma and sentence-transformer embeddings are optional. Owner corrections have a separate data path from general examples. Candidate evaluation considers relevance, style, risk, repeated questions and consistency with the current conversation.

The current pipeline contains conversation-specific heuristics and repair paths. They are inspectable, but can overfit particular phrasing. The public example profile uses fictional names and places; it is not an assessment of the owner's actual communication style.

## Permissions and memory

Contact profiles, automation allowlists, relationship restrictions, duplicate state, rate limits and emergency stop determine whether an action may proceed. Review mode is the default. A generated candidate and an approved device action are different records.

Runtime state includes SQLite logs, conversation memory, corrections, screenshots and vector indexes. Those files can contain sensitive data even if the code does not. The public repository includes source, synthetic demo scenarios, text test fixtures and two generic perception/template configuration files.

## Recording demo

`apps/demo/main.py` creates a temporary data directory for each server lifetime, seeds three fictional scenarios and constructs `DraftingService` with model generation and external APIs disabled. It does not import the live controller, load `.env`, or instantiate ADB.

The demo API accepts scenario identifiers, runs the drafting library and returns candidate scores. Approval is recorded in memory. There is no send implementation; `/api/send` always returns 403. Reset clears local draft approvals. Closing the server removes its temporary sample state.

## Tradeoffs and next work

- ADB is practical for a local prototype but depends on screen state, timing and device layouts. Stronger accessibility semantics and broader device fixtures would improve reliability.
- Static pages make the controller straightforward to run, but repeated UI code and the large orchestration services need clearer module boundaries as the project grows.
- Heuristic scores are useful for inspection, not calibrated probabilities. Broader held-out evaluations would be needed before drawing reliability conclusions.
- Local storage simplifies ownership, but the full controller has no authentication layer or multi-user isolation. It belongs on loopback.
- The browser extension relies on WhatsApp Web internals and privileged browser APIs. UI changes require fresh integration testing.
