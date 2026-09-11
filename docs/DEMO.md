# Recording and interview walkthrough

## A 45-60 second recording

| Time | Screen action | Talking point |
| --- | --- | --- |
| 0-8s | Open the demo with Making plans selected | "Phone Copilot is a supervised phone assistant. This recording uses fictional conversations." |
| 8-20s | Generate drafts | "Context goes through retrieval and candidate ranking. The score helps compare drafts; it is not permission to send." |
| 20-30s | Open Score breakdown, then approve one reply | "Approval is a separate step. Here it stays local because this is the recording demo." |
| 30-42s | Select Work conversation and generate | "Relationship context is part of the decision. Work and unknown contacts require review." |
| 42-52s | Show the execution trace and disabled delivery status | "The full controller adds ADB observation, policy checks and post-action verification. The demo has no phone connection." |
| 52-60s | Switch to the repository README and architecture diagram | "The source includes the controller, retrieval, provider adapters and tests." |

Record at 1440 x 900 for a desktop walkthrough or 390 x 844 for a portrait clip. On mobile, capture the conversation first and scroll to candidates and the trace. Use a clean browser window without personal tabs or bookmarks. The demo needs no credentials, personal chats or real phone footage.

## A longer technical walkthrough

1. Start with the failure being prevented: stale UI state can make a valid-looking tap act on the wrong control.
2. Open `libs/planners/state_machine.py` and `libs/policies/engine.py` to show typed plans and checks before execution.
3. Open the executor verification tests to explain how a successful command differs from a verified UI result.
4. Follow `libs/drafting/service.py` from context through retrieval to candidate scores. Explain the optional provider boundary and deterministic fallback.
5. Show the recording demo's temporary storage and disabled delivery route. Explain why presentation data and personal data need separate lifecycles.
6. Run the tests and discuss a concrete limitation: device layout variance, uncalibrated scores, or the large controller module.

## Questions to prepare for

- Why use ADB rather than an accessibility service? Discuss prototype speed, installation effort and the cost of coordinate-based control.
- What prevents an incorrect reply from being sent? Explain review, relationship policy, allowlists, duplicate checks and post-action verification separately.
- What is local and what goes to a provider? Explain SQLite and retrieval versus optional external model calls.
- What does the demo establish? It shows real deterministic drafting and local approval. It does not establish LLM quality or live-device reliability.
- What would change for a production release? Authentication, smaller modules, stronger UI semantics, held-out quality evaluations and more device coverage.

Use the repository and current test output as evidence. Keep authorship, performance and production claims consistent with what you can personally explain and demonstrate.
