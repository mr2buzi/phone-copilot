from pathlib import Path
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import Response

from apps.controller.models import (
    ApprovalRequest,
    AutomationModeUpdate,
    BlacklistUpdate,
    CatbotChatRequest,
    CatbotFeedbackRequest,
    ComposeValidationRequest,
    FeedbackRequest,
    NotificationPayload,
    ProviderCompareRequest,
    RegenerateRequest,
    SendAndReadRequest,
    StyleReviewApproveRequest,
    StyleReviewRejectRequest,
    TrainingChatRequest,
    TrainingFeedbackRequest,
    TrainingLoopConfigUpdate,
    TrainingQuarantineRequest,
    ThreadOpenRequest,
    WhatsAppWebDraftRequest,
    WhatsAppWebMemoryRequest,
)
from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings

BASE_DIR = Path(__file__).resolve().parents[2]
UI_DIR = BASE_DIR / "apps" / "desktop_ui"

settings = ControllerSettings()
service = PhoneCopilotService(settings=settings)
app = FastAPI(title="phone-copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", f"http://{settings.host}:{settings.port}"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def state_response(state_obj, request_started: float | None = None) -> dict[str, object]:
    serialize_started = time.perf_counter()
    if hasattr(state_obj, "timings_ms"):
        state_obj.timings_ms["UI/state_serialization"] = 0.0
    payload = state_obj.model_dump(mode="json")
    serialization_ms = (time.perf_counter() - serialize_started) * 1000
    if hasattr(state_obj, "timings_ms"):
        state_obj.timings_ms["UI/state_serialization"] = round(serialization_ms, 2)
        if request_started is not None:
            state_obj.timings_ms["response_return"] = round((time.perf_counter() - request_started) * 1000, 2)
            if state_obj.metrics is not None:
                state_obj.metrics.detailed_timings_ms = {
                    key: float(value)
                    for key, value in state_obj.timings_ms.items()
                    if isinstance(value, (int, float))
                }
        payload = state_obj.model_dump(mode="json")
    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/training")
def training_page() -> FileResponse:
    return FileResponse(UI_DIR / "training.html")


@app.get("/training/simple")
def training_simple_page() -> FileResponse:
    return FileResponse(UI_DIR / "training_simple.html")


@app.get("/catbot")
def catbot_page() -> FileResponse:
    return FileResponse(UI_DIR / "catbot.html")


@app.get("/ai-core")
def ai_core_page() -> FileResponse:
    return FileResponse(UI_DIR / "ai_core.html")


@app.get("/ai-core/vector-map")
def vector_map_page() -> FileResponse:
    return FileResponse(UI_DIR / "vector_map.html")


@app.get("/mission-control")
def mission_control_page() -> FileResponse:
    return FileResponse(UI_DIR / "mission_control.html")


@app.get("/execution-trace")
def execution_trace_page() -> FileResponse:
    return FileResponse(UI_DIR / "execution_trace.html")


@app.get("/signal-replay")
def signal_replay_page() -> FileResponse:
    return FileResponse(UI_DIR / "signal_replay.html")


@app.get("/vision-layer")
def vision_layer_page() -> FileResponse:
    return FileResponse(UI_DIR / "vision_layer.html")


@app.get("/memory-inspector")
def memory_inspector_page() -> FileResponse:
    return FileResponse(UI_DIR / "memory_inspector.html")


@app.get("/contact-graph")
def contact_graph_page() -> FileResponse:
    return FileResponse(UI_DIR / "contact_graph.html")


@app.get("/failure-atlas")
def failure_atlas_page() -> FileResponse:
    return FileResponse(UI_DIR / "failure_atlas.html")


@app.get("/risk-engine")
def risk_engine_page() -> FileResponse:
    return FileResponse(UI_DIR / "risk_engine.html")


@app.get("/memory-forge")
def memory_forge_page() -> FileResponse:
    return FileResponse(UI_DIR / "memory_forge.html")


@app.get("/simulation-sandbox")
def simulation_sandbox_page() -> FileResponse:
    return FileResponse(UI_DIR / "simulation_sandbox.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(UI_DIR / "styles.css")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(UI_DIR / "app.js", media_type="application/javascript")


@app.get("/training.js")
def training_js() -> FileResponse:
    return FileResponse(UI_DIR / "training.js", media_type="application/javascript")


@app.get("/training_simple.js")
def training_simple_js() -> FileResponse:
    return FileResponse(UI_DIR / "training_simple.js", media_type="application/javascript")


@app.get("/catbot.js")
def catbot_js() -> FileResponse:
    return FileResponse(UI_DIR / "catbot.js", media_type="application/javascript")


@app.get("/ai_core.js")
def ai_core_js() -> FileResponse:
    return FileResponse(UI_DIR / "ai_core.js", media_type="application/javascript")


@app.get("/vector_map.js")
def vector_map_js() -> FileResponse:
    return FileResponse(UI_DIR / "vector_map.js", media_type="application/javascript")


@app.get("/cyber.css")
def cyber_css() -> FileResponse:
    return FileResponse(UI_DIR / "cyber.css")


@app.get("/cyber_shell.js")
def cyber_shell_js() -> FileResponse:
    return FileResponse(UI_DIR / "cyber_shell.js", media_type="application/javascript")


@app.get("/mission_control.js")
def mission_control_js() -> FileResponse:
    return FileResponse(UI_DIR / "mission_control.js", media_type="application/javascript")


@app.get("/execution_trace.js")
def execution_trace_js() -> FileResponse:
    return FileResponse(UI_DIR / "execution_trace.js", media_type="application/javascript")


@app.get("/signal_replay.js")
def signal_replay_js() -> FileResponse:
    return FileResponse(UI_DIR / "signal_replay.js", media_type="application/javascript")


@app.get("/vision_layer.js")
def vision_layer_js() -> FileResponse:
    return FileResponse(UI_DIR / "vision_layer.js", media_type="application/javascript")


@app.get("/memory_inspector.js")
def memory_inspector_js() -> FileResponse:
    return FileResponse(UI_DIR / "memory_inspector.js", media_type="application/javascript")


@app.get("/contact_graph.js")
def contact_graph_js() -> FileResponse:
    return FileResponse(UI_DIR / "contact_graph.js", media_type="application/javascript")


@app.get("/failure_atlas.js")
def failure_atlas_js() -> FileResponse:
    return FileResponse(UI_DIR / "failure_atlas.js", media_type="application/javascript")


@app.get("/risk_engine.js")
def risk_engine_js() -> FileResponse:
    return FileResponse(UI_DIR / "risk_engine.js", media_type="application/javascript")


@app.get("/memory_forge.js")
def memory_forge_js() -> FileResponse:
    return FileResponse(UI_DIR / "memory_forge.js", media_type="application/javascript")


@app.get("/simulation_sandbox.js")
def simulation_sandbox_js() -> FileResponse:
    return FileResponse(UI_DIR / "simulation_sandbox.js", media_type="application/javascript")


@app.get("/api/media/{category}/{filename}")
def media_file(category: str, filename: str) -> FileResponse:
    roots = {
        "screenshot": service.settings.screenshot_dir.resolve(),
        "debug": service.settings.debug_dir.resolve(),
    }
    root = roots.get(category)
    if root is None:
        raise HTTPException(status_code=404, detail="Unknown media category.")
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Invalid media path.") from exc
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Media file not found.")
    return FileResponse(candidate)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def get_state() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.refresh_state(), started)


@app.get("/api/mission-control/status")
def mission_control_status() -> dict[str, object]:
    return service.cyber_mission_control_status()


@app.get("/api/execution-trace/latest")
def execution_trace_latest() -> dict[str, object]:
    return service.cyber_execution_trace_latest()


@app.get("/api/signal-replay/runs")
def signal_replay_runs(limit: int = 25) -> dict[str, object]:
    return service.cyber_signal_replay_runs(limit=limit)


@app.get("/api/signal-replay/runs/{run_id}")
def signal_replay_run(run_id: str) -> dict[str, object]:
    return service.cyber_signal_replay_run(run_id)


@app.get("/api/vision-layer/latest")
def vision_layer_latest() -> dict[str, object]:
    return service.cyber_vision_layer_latest()


@app.get("/api/memory-inspector/data")
def memory_inspector_data(limit: int = 500) -> dict[str, object]:
    return service.cyber_memory_inspector_data(limit=limit)


@app.get("/api/memory-inspector/example/{example_id}")
def memory_inspector_example(example_id: str) -> dict[str, object]:
    return service.cyber_memory_inspector_example(example_id)


@app.post("/api/memory-inspector/retrieval-battle")
def memory_inspector_retrieval_battle(payload: dict[str, object]) -> dict[str, object]:
    return service.cyber_retrieval_battle(payload)


@app.get("/api/contact-graph/data")
def contact_graph_data() -> dict[str, object]:
    return service.cyber_contact_graph_data()


@app.get("/api/failure-atlas/stats")
def failure_atlas_stats() -> dict[str, object]:
    return service.cyber_failure_atlas_stats()


@app.get("/api/risk-engine/status")
def risk_engine_status() -> dict[str, object]:
    return service.cyber_risk_engine_status()


@app.get("/api/memory-forge/status")
def memory_forge_status() -> dict[str, object]:
    return service.cyber_memory_forge_status()


@app.post("/api/simulation/run")
def simulation_run(payload: dict[str, object]) -> dict[str, object]:
    return service.cyber_simulation_run(payload)


@app.post("/api/refresh-state")
def refresh_state() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.refresh_state(), started)


@app.post("/api/approve/type-only")
def approve_type_only(request: ApprovalRequest) -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.approve_type_only(request.suggestion_index), started)


@app.post("/api/approve/ai-draft-to-compose")
def approve_ai_draft_to_compose(request: ApprovalRequest) -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.approve_ai_draft_to_compose(request.suggestion_index), started)


@app.post("/api/approve/send-draft")
def approve_send_draft() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.approve_send_current_draft(), started)


@app.post("/api/validate/compose-only")
def validate_compose_only(request: ComposeValidationRequest) -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.run_compose_validation(request.text), started)


@app.post("/api/validate/send-and-read")
def validate_send_and_read(request: SendAndReadRequest) -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.run_send_and_read(
        request.text,
        wait_timeout_seconds=request.wait_timeout_seconds,
    ), started)


@app.post("/api/approve/open-gallery")
def approve_open_gallery() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.approve_open_gallery(), started)


@app.post("/api/approve/select-photo")
def approve_select_photo(request: ApprovalRequest) -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.approve_select_photo(request.photo_id), started)


@app.post("/api/reject")
def reject() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.reject(), started)


@app.post("/api/emergency-stop")
def emergency_stop() -> dict[str, object]:
    started = time.perf_counter()
    return state_response(service.emergency_stop(), started)


# Automation endpoints
@app.post("/api/automation/set-mode")
def set_automation_mode(request: AutomationModeUpdate) -> dict[str, object]:
    started = time.perf_counter()
    try:
        service.set_automation_mode(request.mode, request.confidence)
        return state_response(service.refresh_state(), started)
    except Exception as e:
        current = service.refresh_state()
        current.last_action = "Set Mode"
        current.last_action_result = str(e)
        return state_response(current, started)


@app.post("/api/automation/scan-inbox")
def scan_inbox() -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.scan_inbox(), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Scan Inbox", "last_action_result": str(e)}


@app.post("/api/automation/open-thread")
def open_thread(request: ThreadOpenRequest) -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.open_thread(request.contact_name), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Open Thread"}


@app.post("/api/automation/scroll-for-context")
def scroll_for_context() -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.scroll_for_context(), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Scroll for Context"}


@app.post("/api/automation/auto-generate-draft")
def auto_generate_draft() -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.auto_generate_draft(), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Auto Generate Draft"}


@app.post("/api/automation/auto-send-reply")
def auto_send_reply(confidence_override: float | None = None) -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.auto_send_reply(confidence_override), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Auto Send Reply"}


@app.post("/api/automation/blacklist/add")
def blacklist_add(request: BlacklistUpdate) -> dict[str, object]:
    try:
        return service.add_blacklist_contact(request.contact_name or "")
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/automation/blacklist/remove")
def blacklist_remove(request: BlacklistUpdate) -> dict[str, object]:
    try:
        return service.remove_blacklist_contact(request.contact_name or "")
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/automation/state")
def get_automation_state() -> dict[str, object]:
    try:
        return service.get_automation_state()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/debug/timings")
def debug_timings() -> dict[str, object]:
    return service.get_debug_timings()


@app.post("/api/automation/process-queue")
def process_queue_item() -> dict[str, object]:
    started = time.perf_counter()
    try:
        return state_response(service.process_queue_item(), started)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Process Queue"}


@app.post("/api/automation/run-target-cycle")
def run_target_cycle(dry_run: bool = True) -> dict[str, object]:
    try:
        return service.run_target_cycle(dry_run=dry_run)
    except Exception as e:
        return {"status": "error", "error": str(e), "last_action": "Run Target Cycle"}


@app.post("/api/automation/feedback")
def automation_feedback(request: FeedbackRequest) -> dict[str, object]:
    try:
        return service.record_feedback(request.model_dump(mode="json"))
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/automation/regenerate")
def automation_regenerate(request: RegenerateRequest) -> dict[str, object]:
    try:
        bundle = service.drafting.regenerate_bundle(**request.model_dump(mode="json"))
        return bundle.model_dump(mode="json")
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/training/chat")
def training_chat(request: TrainingChatRequest) -> dict[str, object]:
    return service.training_chat(request)


@app.post("/api/training/regenerate")
def training_regenerate(request: TrainingChatRequest) -> dict[str, object]:
    return service.training_regenerate(request)


@app.post("/api/training/catbot/chat")
def training_catbot_chat(request: CatbotChatRequest) -> dict[str, object]:
    return service.training_catbot_chat(request)


@app.post("/api/web/whatsapp/draft")
def whatsapp_web_draft(request: WhatsAppWebDraftRequest) -> dict[str, object]:
    return service.whatsapp_web_draft(request)


@app.post("/api/web/whatsapp/memory")
def whatsapp_web_memory(request: WhatsAppWebMemoryRequest) -> dict[str, object]:
    return service.whatsapp_web_memory_store(request)


@app.post("/api/training/catbot/feedback")
def training_catbot_feedback(request: CatbotFeedbackRequest) -> dict[str, object]:
    return service.training_catbot_feedback(request)


@app.post("/api/training/feedback")
def training_feedback(request: TrainingFeedbackRequest) -> dict[str, object]:
    return service.training_feedback(request)


@app.get("/api/training/style-review-queue")
def training_style_review_queue() -> dict[str, object]:
    return service.training_style_review_queue()


@app.post("/api/training/style-review-queue/approve")
def training_style_review_approve(request: StyleReviewApproveRequest) -> dict[str, object]:
    return service.training_style_review_approve(request)


@app.post("/api/training/style-review-queue/reject")
def training_style_review_reject(request: StyleReviewRejectRequest) -> dict[str, object]:
    return service.training_style_review_reject(request)


@app.get("/api/training/stats")
def training_stats() -> dict[str, object]:
    return service.training_stats()


@app.get("/api/training/training-status")
def training_control_status() -> dict[str, object]:
    return service.training_status()


@app.get("/api/training/loop/status")
def training_loop_status() -> dict[str, object]:
    return service.training_loop_status()


@app.post("/api/training/loop/start")
def training_loop_start() -> dict[str, object]:
    return service.training_loop_start()


@app.post("/api/training/loop/stop")
def training_loop_stop() -> dict[str, object]:
    return service.training_loop_stop()


@app.post("/api/training/loop/run-now")
def training_loop_run_now() -> dict[str, object]:
    return service.training_loop_run_now()


@app.post("/api/training/loop/config")
def training_loop_config(request: TrainingLoopConfigUpdate) -> dict[str, object]:
    return service.training_loop_update_config(request.model_dump(mode="json", exclude_none=True))


@app.post("/api/training/run-evaluation")
def training_run_evaluation(limit: int | None = None) -> dict[str, object]:
    return service.training_run_evaluation(limit=limit)


@app.post("/api/training/generate-improvements")
def training_generate_improvements(limit: int | None = None) -> dict[str, object]:
    return service.training_generate_improvements(limit=limit, write=True)


@app.post("/api/training/run-iteration")
def training_run_iteration(limit: int | None = None) -> dict[str, object]:
    return service.training_run_iteration(limit=limit)


@app.get("/api/training/active-learning-queue")
def training_active_learning_queue() -> dict[str, object]:
    return service.training_active_learning_queue()


@app.get("/api/ai-core/status")
def ai_core_status() -> dict[str, object]:
    return service.ai_core_status()


@app.post("/api/ai-core/reload-identity-pack")
def ai_core_reload_identity_pack() -> dict[str, object]:
    return service.ai_core_reload_identity_pack()


@app.post("/api/ai-core/provider-compare")
def ai_core_provider_compare(request: ProviderCompareRequest) -> dict[str, object]:
    return service.ai_core_provider_compare(request)


@app.get("/api/ai-core/vector-map")
def ai_core_vector_map(limit: int = 900, memory_type: str = "all") -> dict[str, object]:
    return service.ai_core_vector_map(limit=limit, memory_type=memory_type)


@app.post("/api/training/rebuild-vector-db")
def training_rebuild_vector_db() -> dict[str, object]:
    return service.training_rebuild_vector_db()


@app.get("/api/training/recent")
def training_recent(limit: int = 50) -> dict[str, object]:
    return service.training_recent(limit=limit)


@app.post("/api/training/quarantine-example")
def training_quarantine_example(request: TrainingQuarantineRequest) -> dict[str, object]:
    return service.training_quarantine_example(request)


@app.post("/api/notifications")
def notifications(payload: NotificationPayload) -> dict[str, str]:
    service.handle_notification(payload)
    return {"status": "accepted"}


@app.get("/api/latest-screenshot", response_model=None)
def latest_screenshot():
    try:
        screenshot_path = service.latest_screenshot_path
        if screenshot_path is None or not screenshot_path.exists():
            # Try to find any recent screenshot
            import glob
            screenshots = sorted(glob.glob(str(settings.screenshot_dir / "*.png")), reverse=True)
            if not screenshots:
                return {"status": "no_screenshot", "message": "No trace recorded yet"}
            screenshot_path = Path(screenshots[0])
        return FileResponse(screenshot_path, media_type="image/png")
    except Exception as e:
        return {"status": "error", "message": "No trace recorded yet"}


@app.get("/api/logs")
def logs() -> dict[str, object]:
    return {"items": service.recent_logs()}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return service.metrics_snapshot().model_dump(mode="json")


@app.get("/last_state")
def last_state() -> dict[str, object] | None:
    state = service.last_state()
    return None if state is None else state.model_dump(mode="json")


if __name__ == "__main__":
    uvicorn.run("apps.controller.main:app", host=settings.host, port=settings.port, reload=False)
