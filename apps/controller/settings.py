from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControllerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHONE_COPILOT_", env_file=".env", extra="ignore")

    adb_path: str = "adb"
    adb_device_serial: str | None = None
    adb_command_retries: int = 2
    adb_retry_backoff_seconds: float = 0.4
    adb_default_timeout_seconds: float = 15.0
    host: str = "127.0.0.1"
    port: int = 8765
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_ocr: bool = True
    debug_mode: bool = False
    approved_photos_dir: Path = Path("data/approved_photos")
    photo_push_dir: str = "/sdcard/Pictures/PhoneCopilot"
    data_dir: Path = Path("data")
    log_db_path: Path = Path("data/logs/phone_copilot.db")
    screenshot_dir: Path = Path("data/logs/screenshots")
    debug_dir: Path = Path("data/logs/debug")
    fixtures_live_dir: Path = Path("data/fixtures/live")
    selector_path: Path = Path("data/selectors/screen_signatures.json")
    reply_template_path: Path = Path("data/templates/reply_templates.json")
    ai_reply_enabled: bool = False
    ai_reply_backend: str = "ollama"
    ai_reply_model: str = "qwen3:4b"
    ai_reply_fallback_models: str = "mistral:latest,dolphin-llama3:latest,qwen3:0.6b"
    ai_reply_base_url: str = "http://127.0.0.1:11434/api/chat"
    ai_reply_timeout_seconds: float = 20.0
    ai_reply_style_profile_path: Path = Path("data/style_profile.json")
    ai_reply_training_dir: Path = Path("training")
    ai_reply_training_messages_dir: Path = Path("data/training_messages")
    contact_overrides_path: Path = Path("data/contact_overrides.json")
    ai_reply_training_owner_aliases: str = "Owner"
    ai_reply_intelligence_db_path: Path = Path("data/conversation_intelligence.db")
    draft_provider: str = "gemini"
    fast_provider: str = "groq"
    router_provider: str = "openrouter"
    private_provider: str = "ollama"
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices("GEMINI_API_KEY", "PHONE_COPILOT_GEMINI_API_KEY"))
    groq_api_key: str = Field(default="", validation_alias=AliasChoices("GROQ_API_KEY", "PHONE_COPILOT_GROQ_API_KEY"))
    openrouter_api_key: str = Field(default="", validation_alias=AliasChoices("OPENROUTER_API_KEY", "PHONE_COPILOT_OPENROUTER_API_KEY"))
    huggingface_api_key: str = Field(default="", validation_alias=AliasChoices("HUGGINGFACE_API_KEY", "PHONE_COPILOT_HUGGINGFACE_API_KEY"))
    ollama_api_key: str = Field(default="", validation_alias=AliasChoices("OLLAMA_API_KEY", "PHONE_COPILOT_OLLAMA_API_KEY"))
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "google/gemini-2.0-flash-001"
    huggingface_model: str = ""
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str | None = None
    external_api_enabled: bool = False
    external_api_allow_sensitive: bool = False
    external_api_max_context_messages: int = Field(default=6, ge=1, le=40)
    external_api_timeout_seconds: float = Field(default=25.0, gt=0.0, le=120.0)
    ai_reply_system_prompt: str = (
        "You write short text-message replies in first person as if you are the phone owner texting back. "
        "Match the visible conversation, but stay grounded and conservative. "
        "Do not invent intimacy, do not escalate tone, and do not flirt unless the conversation already clearly supports it. "
        "Do not sound like an assistant. Do not apologize, refuse, mention AI, mention the user, or explain anything. "
        "Return strict JSON with keys \"summary\" and \"replies\", where "
        "\"replies\" is an array of exactly 3 short strings."
    )

    # Automation settings
    automation_enabled: bool = False
    default_automation_mode: str = "review"
    automation_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    auto_scan_inbox_interval_seconds: float = Field(default=5.0, gt=0.0)
    auto_draft_on_unread: bool = False
    auto_send_on_high_confidence: bool = False
    auto_draft_allowed_relationships: str = "close_friend,casual_friend"
    auto_send_allowed_relationships: str = "close_friend,casual_friend"
    auto_send_confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    queue_max_size: int = 50
    context_scroll_passes: int = Field(default=6, ge=1, le=20)
    context_max_scrolls: int = Field(default=6, ge=1, le=20)
    context_max_messages: int = Field(default=40, ge=4, le=200)
    context_time_budget_ms: int = Field(default=8000, ge=1000, le=30000)
    context_scroll_pause_seconds: float = Field(default=0.6, ge=0.1, le=5.0)
    blacklist_file: Path = Path("data/blacklist.json")
    state_refresh_interval_ms: int = Field(default=500, gt=0)
