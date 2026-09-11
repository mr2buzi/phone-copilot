# Release validation

Checked on Windows with Python 3.10 and a Chromium browser on 11 September 2026.

| Check | Result |
| --- | --- |
| Full Python suite | 588 passed, 29 expected failures |
| Original simulator baseline | The 29 expected failures reproduce in an isolated copy of the original source |
| Demo API | All three scenarios produce candidates; approvals stay local; sending is rejected |
| Demo isolation | No ADB construction or provider requests; local environment and live configuration are not loaded |
| Browser workflow | Desktop 1440 x 1000 and portrait 390 x 844; scenarios, approval and reset pass |
| Browser layout | Screenshots reviewed; no horizontal overflow or browser console errors |
| Score display | Raw ranking score shown; no percentage or probability claim |
| Public content checks | Staged source scanned for prohibited files and common credentials; local private-identifier comparison found no remaining matches |
| Python packaging | Wheel build checked locally |

The 29 expected failures are unresolved simulator regressions, not successful checks. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) explains their scope and how to run them without expected-failure annotations. No failures outside that recorded list are accepted by the test command.

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
