# Known simulator regressions

The source baseline has 29 reproducible failures in the advanced conversation simulator tests. They are listed by exact test ID in `tests/known_regressions.json` and marked as strict expected failures. An unexpected pass fails CI so the entry can be reviewed and removed. These are unresolved behaviors, not passing tests.

The affected cases cover provider failure handling, repair-path metadata, repeated conversation shapes, fresh-session reply choices, topic changes and conversation-plan expectations. The current implementation sometimes takes a deterministic path where an older test expects a provider retry or a different repair result. Some response-shape expectations also disagree with the implementation.

The recording demo uses its own isolated deterministic workflow. Phone execution, policy tests and the new demo checks remain outside this expected-failure list.

To see the failures without the annotations:

```sh
python -m pytest tests/test_training_page.py --runxfail -q
```

Resolving these cases needs a separate decision about intended simulator behavior, followed by implementation and evaluation changes. They were not relabelled as successful validation for the public release.
