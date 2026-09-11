from __future__ import annotations

from pathlib import Path

from libs.drafting import ApprovedPhoto
from libs.planners.models import ExecutionPlan, ExecutionStep, PlannerDecision
from libs.screen_states import ApprovalActionType, ScreenClassification, ScreenName


class StateMachinePlanner:
    def __init__(self, photo_push_dir: str) -> None:
        self.photo_push_dir = photo_push_dir.rstrip("/")

    def decide(self, classification: ScreenClassification) -> PlannerDecision:
        if classification.screen == ScreenName.PERMISSION_POPUP:
            return PlannerDecision(
                status="halt",
                reason="Permission prompt detected. Manual handling is required before any action.",
                allowed_actions=[],
            )
        if classification.screen == ScreenName.UNKNOWN_SCREEN:
            return PlannerDecision(
                status="halt",
                reason="Screen classifier could not map the current UI to a supported state.",
                allowed_actions=[],
            )
        if classification.screen == ScreenName.THREAD_VIEW:
            return PlannerDecision(
                status="review",
                reason="Thread view detected. Draft, send, and gallery-open actions are available.",
                allowed_actions=[
                    ApprovalActionType.TYPE_DRAFT,
                    ApprovalActionType.SEND_MESSAGE,
                    ApprovalActionType.OPEN_GALLERY,
                ],
            )
        if classification.screen == ScreenName.GALLERY_PICKER:
            return PlannerDecision(
                status="review",
                reason="Gallery picker detected. Approved photo selection is available.",
                allowed_actions=[ApprovalActionType.SELECT_APPROVED_PHOTO],
            )
        if classification.screen == ScreenName.APP_INBOX:
            return PlannerDecision(
                status="review",
                reason="Inbox detected. Review is allowed but no bounded automation is exposed yet.",
                allowed_actions=[],
            )
        return PlannerDecision(
            status="review",
            reason="Home screen detected. No outbound action is proposed.",
            allowed_actions=[],
        )

    def build_plan(
        self,
        classification: ScreenClassification,
        action_type: ApprovalActionType,
        draft_text: str | None = None,
        approved_photo: ApprovedPhoto | None = None,
    ) -> ExecutionPlan:
        action_lookup = {action.action_type: action for action in classification.available_actions}
        if action_type not in action_lookup and action_type != ApprovalActionType.SELECT_APPROVED_PHOTO:
            raise ValueError(
                f"Unsupported planner request {action_type} for screen {classification.screen.value}."
            )
        if action_type == ApprovalActionType.TYPE_DRAFT:
            action = action_lookup[action_type]
            return ExecutionPlan(
                requested_action=action_type,
                expected_prev_screen=ScreenName.THREAD_VIEW,
                expected_next_screen=ScreenName.THREAD_VIEW,
                acceptable_next_screens=[ScreenName.THREAD_VIEW],
                allowed_package_names=[classification.package_name],
                timeout_ms=2500,
                retry=2,
                expected_keyboard_before=classification.keyboard_visible if not classification.keyboard_ambiguous else None,
                expected_keyboard_after=True,
                steps=[
                    ExecutionStep(
                        kind="tap",
                        x=action.point.x,
                        y=action.point.y,
                        region_left=action.region.left if action.region else None,
                        region_top=action.region.top if action.region else None,
                        region_right=action.region.right if action.region else None,
                        region_bottom=action.region.bottom if action.region else None,
                    ),
                    ExecutionStep(kind="type_text", text=draft_text or ""),
                ],
                notes="Type the approved draft into the compose field without sending.",
            )
        if action_type == ApprovalActionType.OPEN_GALLERY:
            action = action_lookup[action_type]
            return ExecutionPlan(
                requested_action=action_type,
                expected_prev_screen=ScreenName.THREAD_VIEW,
                expected_next_screen=ScreenName.GALLERY_PICKER,
                acceptable_next_screens=[ScreenName.GALLERY_PICKER],
                allowed_package_names=[classification.package_name],
                timeout_ms=2500,
                retry=2,
                expected_keyboard_before=False,
                expected_keyboard_after=False,
                steps=[
                    ExecutionStep(
                        kind="tap",
                        x=action.point.x,
                        y=action.point.y,
                        region_left=action.region.left if action.region else None,
                        region_top=action.region.top if action.region else None,
                        region_right=action.region.right if action.region else None,
                        region_bottom=action.region.bottom if action.region else None,
                    )
                ],
                notes="Open the gallery picker from the thread view.",
            )
        if action_type == ApprovalActionType.SEND_MESSAGE:
            action = action_lookup[action_type]
            return ExecutionPlan(
                requested_action=action_type,
                expected_prev_screen=ScreenName.THREAD_VIEW,
                expected_next_screen=ScreenName.THREAD_VIEW,
                acceptable_next_screens=[ScreenName.THREAD_VIEW],
                allowed_package_names=[classification.package_name],
                timeout_ms=3500,
                retry=1,
                expected_keyboard_before=True if not classification.keyboard_ambiguous else None,
                expected_keyboard_after=None,
                steps=[
                    ExecutionStep(
                        kind="tap",
                        x=action.point.x,
                        y=action.point.y,
                        region_left=action.region.left if action.region else None,
                        region_top=action.region.top if action.region else None,
                        region_right=action.region.right if action.region else None,
                        region_bottom=action.region.bottom if action.region else None,
                    )
                ],
                notes="Send the current composed draft from the active thread.",
            )
        if action_type == ApprovalActionType.SELECT_APPROVED_PHOTO and approved_photo is not None:
            if action_type not in action_lookup:
                raise ValueError(
                    f"Unsupported planner request {action_type} for screen {classification.screen.value}."
                )
            action = action_lookup[action_type]
            photo_path = Path(approved_photo.path)
            remote_path = f"{self.photo_push_dir}/{photo_path.name}"
            return ExecutionPlan(
                requested_action=action_type,
                expected_prev_screen=ScreenName.GALLERY_PICKER,
                expected_next_screen=ScreenName.THREAD_VIEW,
                acceptable_next_screens=[ScreenName.THREAD_VIEW, ScreenName.GALLERY_PICKER],
                allowed_package_names=[classification.package_name],
                timeout_ms=3000,
                retry=2,
                expected_keyboard_before=False,
                expected_keyboard_after=False,
                steps=[
                    ExecutionStep(kind="push_file", local_path=str(photo_path), remote_path=remote_path),
                    ExecutionStep(kind="media_scan", remote_path=remote_path),
                    ExecutionStep(
                        kind="tap",
                        x=action.point.x,
                        y=action.point.y,
                        region_left=action.region.left if action.region else None,
                        region_top=action.region.top if action.region else None,
                        region_right=action.region.right if action.region else None,
                        region_bottom=action.region.bottom if action.region else None,
                    ),
                ],
                notes="Push the approved photo to the device and select the newest tile in the picker.",
            )
        raise ValueError(f"Unsupported planner request {action_type} for screen {classification.screen.value}.")
