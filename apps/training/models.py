from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


class ImportedMessage(BaseModel):
    timestamp: datetime | None = None
    sender: str
    text: str
    sender_me: bool


class ImportedChat(BaseModel):
    contact_name: str
    source_path: str
    messages: list[ImportedMessage] = Field(default_factory=list)


class ReplyExampleRecord(BaseModel):
    contact_name: str
    persona: str
    incoming_context: list[str] = Field(default_factory=list)
    my_previous_style_window: list[str] = Field(default_factory=list)
    target_reply: list[str] = Field(default_factory=list)
    source_path: str
    timestamp: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    quality_score: float = 1.0


class RetrievedReplyExample(BaseModel):
    contact_name: str
    persona: str
    incoming_context: list[str] = Field(default_factory=list)
    target_reply: list[str] = Field(default_factory=list)
    score: float = 0.0
    source_path: str = ""


class ContactInsight(BaseModel):
    contact_name: str
    persona: str
    style_profile: dict[str, object] = Field(default_factory=dict)


class TrainingContact(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    display_name: str = SQLField(index=True, unique=True)
    normalized_name: str = SQLField(index=True, unique=True)
    persona: str = SQLField(default="casual_friend", index=True)
    blacklist: bool = False
    auto_send_allowed: bool = False


class TrainingMessage(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    contact_id: int = SQLField(index=True)
    timestamp: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc), index=True)
    sender_me: bool = SQLField(index=True)
    body: str
    source: str = "whatsapp_import"
    quality_score: float = 1.0


class TrainingReplyExample(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    contact_id: int = SQLField(index=True)
    persona: str = SQLField(index=True)
    timestamp: datetime | None = SQLField(default=None, index=True)
    incoming_context_json: str
    my_previous_style_window_json: str = "[]"
    target_reply_json: str
    metadata_json: str = "{}"
    style_tags_json: str = "[]"
    source_path: str
    quality_score: float = 1.0


class TrainingStyleProfile(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    scope: str = SQLField(index=True)
    scope_key: str = SQLField(index=True)
    profile_json: str
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc), index=True)
