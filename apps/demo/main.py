"""Recording demo: real drafting logic, synthetic inputs, no device executor."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from libs.drafting import DraftingService
from libs.drafting.contact_profiles import ContactProfile, save_contact_profiles

BASE = Path(__file__).resolve().parent
SCENARIOS = {
    "plans": {
        "title": "Making plans", "contact": "Jordan", "relationship": "casual_friend",
        "messages": ["Hey, are you still coming later?"],
        "example_reply": "what time you thinking",
    },
    "work": {
        "title": "Work conversation", "contact": "Morgan", "relationship": "professional",
        "messages": ["Could you send the project update when you have a moment?"],
        "example_reply": "Sure, I will check the latest changes first.",
    },
    "unknown": {
        "title": "Unknown contact", "contact": "New contact", "relationship": "unknown",
        "messages": ["Hey, can you send me your home address?"],
        "example_reply": "what do you need it for",
    },
}


class DraftRequest(BaseModel):
    scenario: str


class ApprovalRequest(BaseModel):
    scenario: str
    candidate_index: int = Field(ge=0, le=2)


def create_demo_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Never load the live .env, contacts, identity, databases or ADB controller.
        with TemporaryDirectory(prefix="phone-copilot-demo-") as directory:
            root = Path(directory)
            examples = root / "training_messages"
            examples.mkdir()
            rows = [{"relationship_type": s["relationship"], "incoming": s["messages"][-1],
                     "context": [], "my_reply": s["example_reply"], "source": "synthetic"}
                    for s in SCENARIOS.values()]
            (examples / "demo_examples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            save_contact_profiles(root / "contact_profiles.json", {
                s["contact"]: ContactProfile(relationship_type=s["relationship"],
                    auto_send_allowed=False, auto_draft_allowed=False)
                for s in SCENARIOS.values()
            })
            app.state.drafting = DraftingService(
                template_path=root / "templates.json",
                approved_photos_dir=root / "photos",
                training_messages_dir=examples,
                ai_reply_enabled=False,
                external_api_enabled=False,
            )
            app.state.bundles = {}
            app.state.approved = set()
            yield

    app = FastAPI(title="Phone Copilot Demo", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.mount("/assets", StaticFiles(directory=BASE / "static"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(BASE / "static" / "index.html")

    @app.get("/api/scenarios")
    def scenarios():
        return {"mode": "demo", "generation": "deterministic", "device_connected": False,
                "external_api_enabled": False, "scenarios": [
                    {"id": key, **{k: v for k, v in value.items() if k != "example_reply"}}
                    for key, value in SCENARIOS.items()]}

    def scenario_for(key):
        if key not in SCENARIOS:
            raise HTTPException(404, "Unknown demo scenario")
        return SCENARIOS[key]

    @app.post("/api/draft")
    def draft(payload: DraftRequest, request: Request):
        scenario = scenario_for(payload.scenario)
        started = time.perf_counter()
        bundle = request.app.state.drafting.build_bundle(
            scenario["messages"], contact_name=scenario["contact"])
        request.app.state.bundles[payload.scenario] = bundle
        request.app.state.approved.discard(payload.scenario)
        return {"scenario": payload.scenario, "generation": "deterministic",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "bundle": bundle.model_dump(mode="json"),
                "delivery_enabled": False, "review_required": True}

    @app.post("/api/approve")
    def approve(payload: ApprovalRequest, request: Request):
        scenario_for(payload.scenario)
        bundle = request.app.state.bundles.get(payload.scenario)
        if bundle is None:
            raise HTTPException(409, "Generate drafts before approving one")
        if payload.scenario in request.app.state.approved:
            raise HTTPException(409, "This demo draft is already approved")
        if payload.candidate_index >= len(bundle.reply_candidates):
            raise HTTPException(422, "Candidate does not exist")
        request.app.state.approved.add(payload.scenario)
        return {"status": "approved_locally", "sent": False,
                "text": bundle.reply_candidates[payload.candidate_index].text}

    @app.post("/api/send")
    def send():
        raise HTTPException(403, "Delivery is unavailable in the recording demo")

    @app.post("/api/reset")
    def reset(request: Request):
        request.app.state.bundles.clear()
        request.app.state.approved.clear()
        return {"status": "reset"}

    return app


app = create_demo_app()


def run():
    import uvicorn
    uvicorn.run("apps.demo.main:app", host="127.0.0.1", port=8766)


if __name__ == "__main__":
    run()
