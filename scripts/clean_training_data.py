from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.drafting.relationships import load_contact_overrides, normalize_contact_name
from libs.drafting.training_data import normalize_relationship_type, row_quality_issues

TRAINING_DIR = Path("data/training_messages")
QUARANTINE_DIR = TRAINING_DIR / "quarantine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--apply", action="store_true", default=False)
    args = parser.parse_args()

    apply_changes = args.apply
    if args.dry_run and args.apply:
        raise SystemExit("Use either --dry-run or --apply, not both.")
    if not args.dry_run and not args.apply:
        apply_changes = False

    overrides_path = args.training_dir.parent / "contact_overrides.json"
    overrides = load_contact_overrides(overrides_path) if overrides_path.exists() else {}
    totals = {"rows": 0, "kept": 0, "quarantined": 0}

    for path in sorted(item for item in args.training_dir.glob("*.jsonl") if item.parent.name != "quarantine"):
        result = clean_file(path, overrides=overrides, apply_changes=apply_changes)
        for key in totals:
            totals[key] += result[key]
        mode = "APPLY" if apply_changes else "DRY-RUN"
        print(
            f"{mode} {path}: rows={result['rows']} kept={result['kept']} "
            f"quarantined={result['quarantined']} reasons={result['reasons']}"
        )

    print(
        f"TOTAL rows={totals['rows']} kept={totals['kept']} "
        f"quarantined={totals['quarantined']} mode={'apply' if apply_changes else 'dry-run'}"
    )


def clean_file(path: Path, *, overrides: dict[str, str], apply_changes: bool) -> dict[str, Any]:
    kept: list[str] = []
    quarantined: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    rows = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues = ["invalid_json"]
            row = {"raw_line": line}
        else:
            if not isinstance(row, dict):
                issues = ["non_object_row"]
                row = {"raw_row": row}
            else:
                issues = row_quality_issues(
                    row,
                    allow_flirty=flirty_allowed(row, overrides),
                    relationship_type=str(row.get("relationship_type", "")),
                )
                issues = [issue for issue in issues if issue != "very_long_reply"]
                if reply_too_long_for_relationship(row):
                    issues.append("reply_too_long")
        issues = list(dict.fromkeys(issues))
        if issues:
            for issue in issues:
                reasons[issue] = reasons.get(issue, 0) + 1
            quarantined.append(
                {
                    "source_file": path.name,
                    "line_number": line_number,
                    "quarantined_at": datetime.now(timezone.utc).isoformat(),
                    "issues": issues,
                    "row": row,
                }
            )
            continue
        kept.append(json.dumps(row, ensure_ascii=True))

    if apply_changes:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        if quarantined:
            quarantine_path = QUARANTINE_DIR / path.name
            with quarantine_path.open("a", encoding="utf-8") as handle:
                for item in quarantined:
                    handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text("".join(item + "\n" for item in kept), encoding="utf-8")
        temp_path.replace(path)

    return {"rows": rows, "kept": len(kept), "quarantined": len(quarantined), "reasons": reasons}


def flirty_allowed(row: dict[str, Any], overrides: dict[str, str]) -> bool:
    if bool(row.get("allow_flirty")):
        return True
    contact = normalize_contact_name(str(row.get("contact_name") or row.get("contact") or ""))
    return bool(contact and overrides.get(contact) == "romantic_interest")


def reply_too_long_for_relationship(row: dict[str, Any]) -> bool:
    relationship = normalize_relationship_type(str(row.get("relationship_type", "")))
    if relationship in {"professional", "university"}:
        return False
    reply = str(row.get("my_reply") or row.get("user_final_reply") or "")
    return len(reply) > 160


if __name__ == "__main__":
    main()
