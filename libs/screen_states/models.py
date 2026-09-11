from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScreenName(str, Enum):
    HOME_SCREEN = "home_screen"
    APP_INBOX = "app_inbox"
    THREAD_VIEW = "thread_view"
    GALLERY_PICKER = "gallery_picker"
    PERMISSION_POPUP = "permission_popup"
    UNKNOWN_SCREEN = "unknown_screen"


class ApprovalActionType(str, Enum):
    TYPE_DRAFT = "type_draft"
    SEND_MESSAGE = "send_message"
    OPEN_GALLERY = "open_gallery"
    SELECT_APPROVED_PHOTO = "select_approved_photo"
    REJECT = "reject"
    EMERGENCY_STOP = "emergency_stop"


class DevicePoint(BaseModel):
    x: int
    y: int


class ScreenRegion(BaseModel):
    left: int
    top: int
    right: int
    bottom: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


class AvailableAction(BaseModel):
    action_type: ApprovalActionType
    label: str
    point: DevicePoint | None = None
    region: ScreenRegion | None = None
    description: str | None = None


class ScreenClassification(BaseModel):
    app: str
    screen: ScreenName
    confidence: float
    visible_text: list[str] = Field(default_factory=list)
    available_actions: list[AvailableAction] = Field(default_factory=list)
    package_name: str
    activity_name: str
    screenshot_width: int
    screenshot_height: int
    recent_messages: list[str] = Field(default_factory=list)
    keyboard_visible: bool | None = None
    keyboard_height: int | None = None
    keyboard_ambiguous: bool = False
    features_used: list[str] = Field(default_factory=list)
    debug_info: dict[str, Any] = Field(default_factory=dict)
