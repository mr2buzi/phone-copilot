from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.screen_states import (
    ApprovalActionType,
    AvailableAction,
    DevicePoint,
    ScreenClassification,
    ScreenName,
    ScreenRegion,
)

MESSAGE_PACKAGES = {
    "com.google.android.apps.messaging": "Google Messages",
    "com.whatsapp": "WhatsApp",
    "com.samsung.android.messaging": "Samsung Messages",
    "com.instagram.android": "Instagram",
}
LAUNCHER_PACKAGES = {
    "com.sec.android.app.launcher",
    "com.google.android.apps.nexuslauncher",
}
GALLERY_PACKAGES = {
    "com.google.android.apps.photos",
    "com.sec.android.gallery3d",
    "com.android.documentsui",
}
SCREEN_THRESHOLDS: dict[ScreenName, float] = {
    ScreenName.HOME_SCREEN: 0.8,
    ScreenName.APP_INBOX: 0.85,
    ScreenName.THREAD_VIEW: 0.70,
    ScreenName.GALLERY_PICKER: 0.8,
    ScreenName.PERMISSION_POPUP: 0.9,
    ScreenName.UNKNOWN_SCREEN: 1.0,
}


class ScreenClassifier:
    def __init__(self, selector_path: Path | None = None) -> None:
        self.signatures: dict[str, Any] = {}
        if selector_path is not None and selector_path.exists():
            self.signatures = json.loads(selector_path.read_text(encoding="utf-8"))

    def classify(
        self,
        package_name: str,
        activity_name: str,
        visible_text: list[str],
        screenshot_width: int,
        screenshot_height: int,
        keyboard_visible: bool | None = None,
        keyboard_height: int | None = None,
        keyboard_ambiguous: bool = False,
        visual_features: dict[str, float] | None = None,
    ) -> ScreenClassification:
        visible_text = [line.strip() for line in visible_text if line.strip()]
        text_blob = " ".join(visible_text).lower()
        features_used: list[str] = []
        debug_scores: dict[str, float] = {}
        visual_features = visual_features or {}

        score_components: dict[str, list[str]] = {}
        for candidate in ScreenName:
            if candidate == ScreenName.UNKNOWN_SCREEN:
                continue
            score, score_features, score_details = self._score_screen(
                candidate=candidate,
                package_name=package_name,
                activity_name=activity_name,
                text_blob=text_blob,
                keyboard_visible=keyboard_visible,
                keyboard_ambiguous=keyboard_ambiguous,
                visual_features=visual_features,
            )
            debug_scores[candidate.value] = round(score, 3)
            score_components[candidate.value] = score_details
            if score_features:
                features_used.extend(score_features)

        top_screen = max(debug_scores, key=debug_scores.get) if debug_scores else ScreenName.UNKNOWN_SCREEN.value
        screen = ScreenName(top_screen)
        confidence = debug_scores.get(screen.value, 0.0)
        threshold = SCREEN_THRESHOLDS.get(screen, 1.0)
        threshold_passed = confidence >= threshold
        fallback_reason: str | None = None
        if not threshold_passed:
            fallback_reason = (
                f"Best candidate {screen.value} scored {confidence:.2f}, below threshold {threshold:.2f}."
            )
            screen = ScreenName.UNKNOWN_SCREEN

        debug_info = {
            "scores": debug_scores,
            "score_components": score_components,
            "thresholds": {name.value: threshold for name, threshold in SCREEN_THRESHOLDS.items()},
            "best_candidate": top_screen,
            "best_candidate_confidence": confidence,
            "threshold_passed": threshold_passed,
            "fallback_reason": fallback_reason,
            "keyboard_visible": keyboard_visible,
            "keyboard_height": keyboard_height,
            "keyboard_ambiguous": keyboard_ambiguous,
            "visual_features": visual_features,
        }

        return ScreenClassification(
            app=MESSAGE_PACKAGES.get(
                package_name,
                package_name.rsplit(".", maxsplit=1)[-1].replace("_", " ").title(),
            ),
            screen=screen,
            confidence=confidence,
            visible_text=visible_text,
            available_actions=self._build_actions(
                screen=screen,
                width=screenshot_width,
                height=screenshot_height,
                keyboard_visible=keyboard_visible,
                keyboard_height=keyboard_height,
            ),
            package_name=package_name,
            activity_name=activity_name,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            recent_messages=self._extract_recent_messages(visible_text),
            keyboard_visible=keyboard_visible,
            keyboard_height=keyboard_height,
            keyboard_ambiguous=keyboard_ambiguous,
            features_used=sorted(set(features_used)),
            debug_info=debug_info,
        )

    def _score_screen(
        self,
        candidate: ScreenName,
        package_name: str,
        activity_name: str,
        text_blob: str,
        keyboard_visible: bool | None,
        keyboard_ambiguous: bool,
        visual_features: dict[str, float],
    ) -> tuple[float, list[str], list[str]]:
        score = 0.0
        features: list[str] = []
        details: list[str] = []
        activity_lower = activity_name.lower()

        if candidate == ScreenName.PERMISSION_POPUP:
            if package_name.startswith("com.android.permissioncontroller"):
                score = self._apply_contribution(score, features, details, 0.7, "permission.package")
            if self._contains_any(text_blob, ["allow", "don't allow", "while using the app", "permission"]):
                score = self._apply_contribution(score, features, details, 0.3, "permission.text")

        elif candidate == ScreenName.HOME_SCREEN:
            if package_name in LAUNCHER_PACKAGES or "launcher" in activity_lower:
                score = self._apply_contribution(score, features, details, 0.75, "home.package")
            if self._contains_any(text_blob, ["search your phone", "finder", "weather", "calendar"]):
                score = self._apply_contribution(score, features, details, 0.15, "home.text")
            bottom_variance = visual_features.get("bottom_band_variance", 0.0)
            center_variance = visual_features.get("center_variance", 0.0)
            if bottom_variance > 4000 and center_variance > 1500:
                score = self._apply_contribution(score, features, details, 0.10, "home.visual")
            elif visual_features.get("top_band_brightness", 0.0) > 180:
                score = self._apply_contribution(score, features, details, 0.05, "home.visual")

        elif candidate == ScreenName.APP_INBOX:
            if package_name in MESSAGE_PACKAGES:
                score = self._apply_contribution(score, features, details, 0.55, "inbox.package")
                if self._is_message_activity(package_name, activity_lower):
                    score = self._apply_contribution(score, features, details, 0.1, "inbox.activity")
            if self._contains_any(text_blob, ["messages", "inbox", "chats", "archived", "search"]):
                score = self._apply_contribution(score, features, details, 0.3, "inbox.text")
            if keyboard_visible is False:
                score = self._apply_contribution(score, features, details, 0.05, "inbox.no_keyboard")
            if keyboard_ambiguous:
                score = self._apply_contribution(score, features, details, -0.02, "inbox.keyboard_ambiguous")

        elif candidate == ScreenName.THREAD_VIEW:
            if package_name in MESSAGE_PACKAGES:
                score = self._apply_contribution(score, features, details, 0.45, "thread.package")
                if self._is_message_activity(package_name, activity_lower):
                    score = self._apply_contribution(score, features, details, 0.08, "thread.activity")
            if self._contains_any(
                text_blob,
                ["type a message", "message", "send", "emoji", "attach", "today", "yesterday"],
            ):
                score = self._apply_contribution(score, features, details, 0.28, "thread.text")
            if package_name in MESSAGE_PACKAGES and "type a message" in text_blob:
                score = self._apply_contribution(score, features, details, 0.12, "thread.compose_hint")
            if keyboard_visible is True:
                score = self._apply_contribution(score, features, details, 0.18, "thread.keyboard")
            if keyboard_ambiguous:
                score = self._apply_contribution(score, features, details, -0.02, "thread.keyboard_ambiguous")
            bottom_variance = visual_features.get("bottom_band_variance", 0.0)
            center_variance = visual_features.get("center_variance", 0.0)
            if bottom_variance > 800 and center_variance > 900:
                score = self._apply_contribution(score, features, details, 0.20, "thread.keyboard_open_visual")
                if keyboard_ambiguous:
                    score = self._apply_contribution(score, features, details, 0.05, "thread.keyboard_ambiguous_visual")
            elif bottom_variance > 600 and center_variance > 600 and keyboard_ambiguous:
                score = self._apply_contribution(score, features, details, 0.25, "thread.keyboard_open_visual")
            elif bottom_variance > 400:
                score = self._apply_contribution(score, features, details, 0.05, "thread.visual")

        elif candidate == ScreenName.GALLERY_PICKER:
            if package_name in GALLERY_PACKAGES:
                score = self._apply_contribution(score, features, details, 0.55, "gallery.package")
            if self._contains_any(text_blob, ["gallery", "albums", "recent", "photos", "camera"]):
                score = self._apply_contribution(score, features, details, 0.25, "gallery.text")
            if keyboard_visible is False:
                score = self._apply_contribution(score, features, details, 0.05, "gallery.no_keyboard")
            if visual_features.get("center_variance", 0.0) > 900:
                score = self._apply_contribution(score, features, details, 0.05, "gallery.visual")

        return min(score, 0.99), features, details

    def _apply_contribution(
        self,
        score: float,
        features: list[str],
        details: list[str],
        amount: float,
        label: str,
    ) -> float:
        if amount:
            score += amount
            features.append(label)
            details.append(f"{label}:{amount:+.2f}")
        return score

    def _is_message_activity(self, package_name: str, activity_lower: str) -> bool:
        if package_name != "com.google.android.apps.messaging":
            return False
        return any(
            pattern in activity_lower
            for pattern in (".main.mainactivity", ".mainactivity", "conversation", "inbox")
        )

    def _extract_recent_messages(self, visible_text: list[str]) -> list[str]:
        ignored_tokens = {
            "send",
            "camera",
            "gallery",
            "emoji",
            "today",
            "yesterday",
            "type a message",
            "search",
            "allow",
            "deny",
        }
        messages = []
        for line in visible_text:
            lowered = line.lower()
            if lowered in ignored_tokens:
                continue
            if any(token in lowered for token in ["type a message", "gallery", "camera", "allow", "photos"]):
                continue
            messages.append(line)
        return messages[-6:]

    def _build_actions(
        self,
        screen: ScreenName,
        width: int,
        height: int,
        keyboard_visible: bool,
        keyboard_height: int,
    ) -> list[AvailableAction]:
        composer_bottom = int(height * 0.94)
        if keyboard_visible and keyboard_height:
            composer_bottom = max(48, height - keyboard_height - 32)
        if screen == ScreenName.THREAD_VIEW:
            return [
                AvailableAction(
                    action_type=ApprovalActionType.TYPE_DRAFT,
                    label="Approve type only",
                    point=DevicePoint(x=int(width * 0.5), y=composer_bottom),
                    region=ScreenRegion(
                        left=int(width * 0.12),
                        top=max(0, composer_bottom - 72),
                        right=int(width * 0.88),
                        bottom=min(height, composer_bottom + 24),
                    ),
                    description="Focus the compose field and type a suggested reply without sending.",
                ),
                AvailableAction(
                    action_type=ApprovalActionType.OPEN_GALLERY,
                    label="Approve open gallery",
                    point=DevicePoint(x=int(width * 0.12), y=composer_bottom),
                    region=ScreenRegion(
                        left=0,
                        top=max(0, composer_bottom - 72),
                        right=int(width * 0.24),
                        bottom=min(height, composer_bottom + 24),
                    ),
                    description="Open the attachment or gallery picker from the current thread.",
                ),
                AvailableAction(
                    action_type=ApprovalActionType.SEND_MESSAGE,
                    label="Approve send message",
                    point=DevicePoint(x=int(width * 0.92), y=composer_bottom),
                    region=ScreenRegion(
                        left=int(width * 0.80),
                        top=max(0, composer_bottom - 72),
                        right=width,
                        bottom=min(height, composer_bottom + 24),
                    ),
                    description="Send the current draft in the active thread.",
                ),
            ]
        if screen == ScreenName.GALLERY_PICKER:
            return [
                AvailableAction(
                    action_type=ApprovalActionType.SELECT_APPROVED_PHOTO,
                    label="Approve select approved photo",
                    point=DevicePoint(x=int(width * 0.22), y=int(height * 0.28)),
                    region=ScreenRegion(
                        left=0,
                        top=int(height * 0.12),
                        right=int(width * 0.45),
                        bottom=int(height * 0.45),
                    ),
                    description="Select the top-left newest media tile after an approved photo is pushed.",
                )
            ]
        return []

    def _contains_any(self, text_blob: str, phrases: list[str]) -> bool:
        return any(phrase in text_blob for phrase in phrases)
