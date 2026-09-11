from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRAINING_LOOP_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "review_only",
    "interval_minutes": 30,
    "max_cases_per_run": 100,
    "max_review_candidates_per_run": 25,
    "auto_rebuild_after_approval": True,
    "auto_approve": False,
    "run_on_startup": False,
    "run_eval": True,
    "generate_improvements": True,
    "build_active_learning_queue": True,
    "rebuild_vector_if_required": True,
    "quiet_hours": {
        "enabled": False,
        "start": "23:00",
        "end": "08:00",
    },
}


class TrainingLoopService:
    def __init__(
        self,
        *,
        data_dir: Path = Path("data"),
        repo_root: Path | None = None,
        rebuild_callback=None,
    ) -> None:
        self.data_dir = data_dir
        self.repo_root = repo_root or Path.cwd()
        self.config_path = self.data_dir / "training_loop_config.json"
        self.status_path = self.data_dir / "reports" / "training_loop_status.json"
        self.review_queue_path = self.data_dir / "training_messages" / "auto_style_corrections_review.jsonl"
        self.active_learning_path = self.data_dir / "training_messages" / "active_learning_queue.jsonl"
        self.rebuild_marker_path = self.data_dir / "vector_db" / "reply_examples_chroma" / ".rebuild_required"
        self.rebuild_callback = rebuild_callback
        self._run_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rebuild_lock = threading.Lock()
        self._rebuild_timer: threading.Timer | None = None
        self._write_default_config_if_missing()
        self._recover_status()

    def load_config(self) -> dict[str, Any]:
        self._write_default_config_if_missing()
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        config = _merge_config(DEFAULT_TRAINING_LOOP_CONFIG, payload if isinstance(payload, dict) else {})
        if bool(config.get("auto_approve")):
            config["auto_approve"] = False
            self._append_warning("auto_approve=true ignored; autonomous training is review-only.")
        config["mode"] = "review_only"
        return config

    def save_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.load_config()
        safe_fields = {
            "enabled",
            "interval_minutes",
            "max_cases_per_run",
            "max_review_candidates_per_run",
            "auto_rebuild_after_approval",
            "run_on_startup",
            "run_eval",
            "generate_improvements",
            "build_active_learning_queue",
            "rebuild_vector_if_required",
            "quiet_hours",
        }
        for key, value in updates.items():
            if key == "auto_approve" and bool(value):
                current["auto_approve"] = False
                self._append_warning("auto_approve=true rejected; generated rows require human approval.")
                continue
            if key not in safe_fields:
                continue
            current[key] = value
        current = _merge_config(DEFAULT_TRAINING_LOOP_CONFIG, current)
        current["mode"] = "review_only"
        current["auto_approve"] = False
        current["interval_minutes"] = _clamp_int(current.get("interval_minutes"), 5, 1440, 30)
        current["max_cases_per_run"] = _clamp_int(current.get("max_cases_per_run"), 1, 1000, 100)
        current["max_review_candidates_per_run"] = _clamp_int(current.get("max_review_candidates_per_run"), 1, 500, 25)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(current, indent=2, ensure_ascii=True), encoding="utf-8")
        self._update_status({"enabled": bool(current["enabled"]), "next_run_at": self._next_run_iso(current)})
        return current

    def status(self) -> dict[str, Any]:
        config = self.load_config()
        status = self._read_status()
        status.update(
            {
                "enabled": bool(config.get("enabled")),
                "running": self._run_lock.locked(),
                "config": config,
                "review_queue_count": self._count_pending_review(),
                "active_learning_queue_count": self._count_jsonl(self.active_learning_path),
                "vector_rebuild_required": self.rebuild_marker_path.exists(),
            }
        )
        if not status.get("next_run_at") and config.get("enabled"):
            status["next_run_at"] = self._next_run_iso(config)
        return status

    def start(self) -> dict[str, Any]:
        config = self.save_config({"enabled": True})
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._loop_forever, name="training-loop", daemon=True)
                self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.save_config({"enabled": False})
        self._stop_event.set()
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        self._update_status({"enabled": False, "next_run_at": None})
        return self.status()

    def run_now(self, *, background: bool = False) -> dict[str, Any]:
        if self._run_lock.locked():
            status = self.status()
            status["status"] = "already_running"
            return status
        if background:
            thread = threading.Thread(target=self.run_once, name="training-loop-run-now", daemon=True)
            thread.start()
            status = self.status()
            status["status"] = "started"
            return status
        result = self.run_once()
        return {**result, "status": "completed", "loop_status": self.status()}

    def run_once(self) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return {"status": "blocked", "reason": "training_loop_already_running"}
        config = self.load_config()
        started_at = _utc_now()
        self._update_status({"running": True, "last_started_at": started_at, "last_error": None})
        try:
            if bool(config.get("auto_approve")):
                config["auto_approve"] = False
                self._append_warning("auto_approve=true ignored during run.")
            result = self._execute_iteration(config)
            finished_at = _utc_now()
            status = self._read_status()
            total_runs = int(status.get("total_runs", 0) or 0) + 1
            self._update_status(
                {
                    "running": False,
                    "last_finished_at": finished_at,
                    "last_result": result,
                    "last_error": None,
                    "total_runs": total_runs,
                    "next_run_at": self._next_run_iso(config),
                }
            )
            return {"status": "ok", **result}
        except Exception as exc:
            self._update_status(
                {
                    "running": False,
                    "last_finished_at": _utc_now(),
                    "last_error": str(exc),
                    "next_run_at": self._next_run_iso(config),
                }
            )
            return {"status": "error", "error": str(exc)}
        finally:
            self._run_lock.release()

    def schedule_rebuild_after_approval(self, *, delay_seconds: float = 7.0) -> None:
        if not self.load_config().get("auto_rebuild_after_approval", True):
            return
        with self._thread_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
            self._rebuild_timer = threading.Timer(delay_seconds, self.rebuild_vector_if_required)
            self._rebuild_timer.daemon = True
            self._rebuild_timer.start()

    def rebuild_vector_if_required(self) -> dict[str, Any]:
        if not self.rebuild_marker_path.exists():
            return {"status": "skipped", "reason": "vector_rebuild_not_required"}
        if not self._rebuild_lock.acquire(blocking=False):
            return {"status": "blocked", "reason": "vector_rebuild_already_running"}
        started = _utc_now()
        try:
            if self.rebuild_callback is not None:
                result = self.rebuild_callback()
            else:
                result = self._run_command([sys.executable, "scripts/build_reply_vector_db.py"])
            if str(result.get("status")) in {"rebuilt", "ok"} or int(result.get("exit_code", 0) or 0) == 0:
                self.rebuild_marker_path.unlink(missing_ok=True)
            else:
                self._mark_rebuild_required("auto_rebuild_failed")
            self._update_status({"last_vector_rebuild_attempt": {"started_at": started, "result": result}})
            return result
        except Exception as exc:
            self._mark_rebuild_required(str(exc))
            self._update_status({"last_vector_rebuild_attempt": {"started_at": started, "error": str(exc)}})
            return {"status": "error", "error": str(exc)}
        finally:
            self._rebuild_lock.release()

    def _execute_iteration(self, config: dict[str, Any]) -> dict[str, Any]:
        max_cases = _clamp_int(config.get("max_cases_per_run"), 1, 1000, 100)
        result: dict[str, Any] = {
            "eval_cases": 0,
            "failures": 0,
            "review_candidates_generated": 0,
            "active_learning_rows": 0,
            "vector_rebuild_run": False,
            "commands": {},
        }
        if config.get("run_eval", True):
            eval_result = self._run_command([sys.executable, "scripts/evaluate_reply_style.py", "--limit", str(max_cases)])
            report = self._read_json(self.data_dir / "reports" / "reply_style_eval_report.json")
            result["eval_cases"] = int(report.get("total_cases", 0) or 0)
            result["failures"] = int(report.get("fail_count", 0) or 0)
            result["commands"]["evaluation"] = eval_result
        if config.get("generate_improvements", True):
            before_pending = self._count_pending_review()
            max_candidates = _clamp_int(config.get("max_review_candidates_per_run"), 1, 500, 25)
            improve = self._run_command(
                [
                    sys.executable,
                    "scripts/improve_failed_replies.py",
                    "--limit",
                    str(max_cases),
                    "--max-write",
                    str(max_candidates),
                    "--write",
                ]
            )
            after_pending = self._count_pending_review()
            parsed = _parse_json_text(str(improve.get("stdout") or ""))
            generated = int(parsed.get("improvements_written", max(0, after_pending - before_pending)) or 0)
            result["review_candidates_generated"] = generated
            result["commands"]["improvements"] = improve
        if config.get("build_active_learning_queue", True):
            active = self._run_command([sys.executable, "scripts/build_active_learning_queue.py"])
            result["active_learning_rows"] = self._count_jsonl(self.active_learning_path)
            result["commands"]["active_learning"] = active
        if config.get("rebuild_vector_if_required", True) and self.rebuild_marker_path.exists():
            rebuild = self.rebuild_vector_if_required()
            result["vector_rebuild_run"] = str(rebuild.get("status")) in {"rebuilt", "ok"}
            result["commands"]["vector_rebuild"] = rebuild
        return result

    def _loop_forever(self) -> None:
        first_cycle = True
        while not self._stop_event.is_set():
            config = self.load_config()
            if not config.get("enabled"):
                break
            sleep_seconds = max(5, int(config.get("interval_minutes", 30) or 30) * 60)
            if first_cycle and not config.get("run_on_startup"):
                self._update_status({"next_run_at": self._next_run_iso(config)})
                first_cycle = False
                self._stop_event.wait(sleep_seconds)
                continue
            if not self._in_quiet_hours(config):
                self.run_once()
            first_cycle = False
            self._update_status({"next_run_at": self._next_run_iso(config)})
            self._stop_event.wait(sleep_seconds)

    def _run_command(self, args: list[str]) -> dict[str, Any]:
        completed = subprocess.run(args, cwd=self.repo_root, text=True, capture_output=True)
        return {
            "status": "ok" if completed.returncode == 0 else "error",
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    def _write_default_config_if_missing(self) -> None:
        if self.config_path.exists():
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(DEFAULT_TRAINING_LOOP_CONFIG, indent=2, ensure_ascii=True), encoding="utf-8")

    def _recover_status(self) -> None:
        status = self._read_status()
        if status.get("running"):
            self._update_status({"running": False, "last_error": "Recovered after restart while previous run was marked running."})
        elif not status:
            self._update_status({"enabled": bool(self.load_config().get("enabled")), "running": False, "total_runs": 0})

    def _read_status(self) -> dict[str, Any]:
        return self._read_json(self.status_path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _update_status(self, patch: dict[str, Any]) -> None:
        status = self._read_status()
        status.update(patch)
        status.setdefault("enabled", False)
        status.setdefault("running", False)
        status.setdefault("last_started_at", None)
        status.setdefault("last_finished_at", None)
        status.setdefault("next_run_at", None)
        status.setdefault("last_result", {})
        status.setdefault("last_error", None)
        status.setdefault("total_runs", 0)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(status, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self.status_path)

    def _append_warning(self, warning: str) -> None:
        status = self._read_status()
        warnings = list(status.get("warnings", [])) if isinstance(status.get("warnings"), list) else []
        warnings.append({"timestamp": _utc_now(), "warning": warning})
        self._update_status({"warnings": warnings[-20:]})

    def _count_pending_review(self) -> int:
        rows = self._read_jsonl(self.review_queue_path)
        return sum(1 for row in rows if str(row.get("status") or "needs_human_review") == "needs_human_review")

    def _count_jsonl(self, path: Path) -> int:
        return len(self._read_jsonl(path))

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _mark_rebuild_required(self, reason: str) -> None:
        self.rebuild_marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.rebuild_marker_path.write_text(
            json.dumps({"timestamp": _utc_now(), "reason": reason}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _next_run_iso(self, config: dict[str, Any]) -> str | None:
        if not config.get("enabled"):
            return None
        return (datetime.now(timezone.utc) + timedelta(minutes=int(config.get("interval_minutes", 30) or 30))).isoformat()

    def _in_quiet_hours(self, config: dict[str, Any]) -> bool:
        quiet = config.get("quiet_hours")
        if not isinstance(quiet, dict) or not quiet.get("enabled"):
            return False
        now = datetime.now().time()
        start = _parse_hhmm(str(quiet.get("start", "23:00")))
        end = _parse_hhmm(str(quiet.get("end", "08:00")))
        if start is None or end is None:
            return False
        if start <= end:
            return start <= now <= end
        return now >= start or now <= end


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_config(defaults: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in payload.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_json_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_hhmm(value: str):
    try:
        hour, minute = value.split(":", 1)
        return datetime.strptime(f"{int(hour):02d}:{int(minute):02d}", "%H:%M").time()
    except Exception:
        return None
