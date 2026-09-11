from pydantic import BaseModel, Field

from libs.screen_states import ApprovalActionType, ScreenName


class ExecutionStep(BaseModel):
    kind: str
    x: int | None = None
    y: int | None = None
    region_left: int | None = None
    region_top: int | None = None
    region_right: int | None = None
    region_bottom: int | None = None
    text: str | None = None
    keycode: int | None = None
    local_path: str | None = None
    remote_path: str | None = None


class ExecutionPlan(BaseModel):
    requested_action: ApprovalActionType
    expected_prev_screen: ScreenName
    expected_next_screen: ScreenName
    acceptable_next_screens: list[ScreenName] = Field(default_factory=list)
    allowed_package_names: list[str] = Field(default_factory=list)
    timeout_ms: int = 2000
    retry: int = 2
    expected_keyboard_before: bool | None = None
    expected_keyboard_after: bool | None = None
    postcondition_text: str | None = None
    steps: list[ExecutionStep] = Field(default_factory=list)
    notes: str


class PlannerDecision(BaseModel):
    status: str
    reason: str
    allowed_actions: list[ApprovalActionType] = Field(default_factory=list)
