# Public data boundary

The public version starts from a new Git history containing reviewed source and fictional examples. Private history is not part of this release.

Excluded material includes environment files, raw chat exports, contact lists, automation targets, personal identity files, training corpora, corrections, vector indexes, SQLite databases, device dumps, screenshots, logs, run reports and local handoff notes.

The code and matching tests use example names and fictional locations. UK telephone fixtures use the reserved `07700 900xxx` range. The sample identity is labelled as fictional, and must be configured before live use. The demo creates temporary synthetic data and never opens the owner's working data directory.

## Before each public commit

1. Review the staged file list and diff.
2. Run `python scripts/check_public_release.py`. It scans the index, so unstaged edits cannot disguise staged content.
3. Review images and documentation manually. Pattern checks cannot detect every private fact or text visible inside a screenshot.
4. Keep runtime data ignored. Do not force-add chat exports, keys or diagnostic captures.

The checker rejects private runtime paths, unexpected binaries, common credential formats, personal Windows paths and non-example UK mobile numbers. GitHub CI runs it on the committed tree. It is a repository-specific check, not a guarantee that every possible secret will be recognized.

If a credential has ever been exposed, revoke it with the provider. Deleting a file or creating a clean public history does not invalidate the credential.
