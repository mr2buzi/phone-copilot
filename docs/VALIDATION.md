# Release validation

## Catbot recovery update, 14 September 2026

Validated on Windows with Python 3.10:

| Check | Result |
| --- | --- |
| Full Python suite | 636 passed; two dependency deprecation warnings |
| Casual follow-up regressions | 12 passed, covering classification and recovery from rejected or unavailable provider output |
| Running controller API | `hi`, `im good wys`, `bruh`, and `fix up` all returned HTTP 200 with nonempty, validated replies |
| Provider bypass | All four API replies used `plan_ranker`, with `external_api_used: false` and `auto_send_allowed: false` |
| Catbot page | `/catbot` returned HTTP 200 |

External requests and automation were disabled for the controller check. No live
model provider or Android conversation was exercised. Existing tests still
require HTTP 502 when neither a provider nor an available local repair produces
a usable reply. See [CATBOT.md](CATBOT.md) for behavior and reproduction steps.

## Recording demo release

Checked on Windows with Python 3.10 and a Chromium browser on 11 September 2026.

| Check | Result |
| --- | --- |
| Full Python suite | All tests passed |
| Simulator regressions | Resolved; the suite runs without expected-failure annotations |
| Demo API | All three scenarios produce candidates; approvals stay local; sending is rejected |
| Demo isolation | No ADB construction or provider requests; local environment and live configuration are not loaded |
| Browser workflow | Desktop 1440 x 1000 and portrait 390 x 844; scenarios, approval and reset pass |
| Browser layout | Screenshots reviewed; no horizontal overflow or browser console errors |
| Score display | Raw ranking score shown; no percentage or probability claim |
| Public content checks | Staged source scanned for prohibited files and common credentials; local private-identifier comparison found no remaining matches |
| Python packaging | Wheel build checked locally |

The simulator tests now run as ordinary tests. No expected-failure list is used by the public checkout.

The demo runs the real deterministic library path, not a live LLM. Candidate scores are heuristic ranking values and can exceed 1; they are not calibrated confidence percentages.

This release was not validated against a real Android device, live WhatsApp conversations or external model providers. Historical private-run results are not presented as evidence for this public snapshot. Linux CI is configured but was not run locally.

## Reproduce

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py --revision HEAD
```

For screenshots, start the demo, then in another terminal:

```sh
npm ci
npx playwright install chromium
node scripts/capture_demo.mjs
```

The screenshot command refreshes `docs/images/demo-desktop.png` and `docs/images/demo-mobile.png` using fictional data. Review both images before committing them.
