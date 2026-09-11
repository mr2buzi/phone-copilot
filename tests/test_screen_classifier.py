import json
from pathlib import Path

from libs.perception import ScreenClassifier
from libs.screen_states import ScreenName


def _load_fixture(name: str) -> dict[str, object]:
    path = Path("tests/fixtures/screens") / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_classifier_detects_thread_view_from_fixture() -> None:
    fixture = _load_fixture("thread_view")
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name=str(fixture["package_name"]),
        activity_name=str(fixture["activity_name"]),
        visible_text=list(fixture["visible_text"]),
        screenshot_width=int(fixture["screenshot_width"]),
        screenshot_height=int(fixture["screenshot_height"]),
    )
    assert classification.screen == ScreenName.THREAD_VIEW
    assert classification.available_actions
    assert "scores" in classification.debug_info


def test_classifier_detects_app_inbox_with_ambiguous_keyboard() -> None:
    fixture = _load_fixture("app_inbox_ambiguous")
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name=str(fixture["package_name"]),
        activity_name=str(fixture["activity_name"]),
        visible_text=list(fixture["visible_text"]),
        screenshot_width=int(fixture["screenshot_width"]),
        screenshot_height=int(fixture["screenshot_height"]),
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
    )
    assert classification.screen == ScreenName.APP_INBOX
    assert classification.confidence >= 0.9
    assert classification.debug_info["score_components"]["app_inbox"]


def test_classifier_detects_thread_view_under_keyboard_ambiguity() -> None:
    fixture = _load_fixture("thread_view_keyboard_open_ambiguous")
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name=str(fixture["package_name"]),
        activity_name=str(fixture["activity_name"]),
        visible_text=list(fixture["visible_text"]),
        screenshot_width=int(fixture["screenshot_width"]),
        screenshot_height=int(fixture["screenshot_height"]),
        keyboard_visible=fixture.get("keyboard_visible"),
        keyboard_height=fixture.get("keyboard_height"),
        keyboard_ambiguous=bool(fixture.get("keyboard_ambiguous", False)),
        visual_features={
            "top_band_brightness": float(fixture["visual_features"]["top_band_brightness"]),
            "bottom_band_variance": float(fixture["visual_features"]["bottom_band_variance"]),
            "center_variance": float(fixture["visual_features"]["center_variance"]),
        },
    )
    assert classification.screen == ScreenName.THREAD_VIEW
    assert classification.confidence >= 0.75
    assert any(
        "thread.keyboard_open_visual" in entry
        for entry in classification.debug_info["score_components"]["thread_view"]
    )


def test_classifier_detects_home_screen_with_launcher_visual_cues() -> None:
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name="com.sec.android.app.launcher",
        activity_name=".activities.LauncherActivity",
        visible_text=[],
        screenshot_width=1080,
        screenshot_height=2340,
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
        visual_features={"top_band_brightness": 25.7, "bottom_band_variance": 4731.24, "center_variance": 1814.85},
    )
    assert classification.screen == ScreenName.HOME_SCREEN
    assert classification.confidence >= 0.80


def test_classifier_detects_permission_popup_from_fixture() -> None:
    fixture = _load_fixture("permission_popup")
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name=str(fixture["package_name"]),
        activity_name=str(fixture["activity_name"]),
        visible_text=list(fixture["visible_text"]),
        screenshot_width=int(fixture["screenshot_width"]),
        screenshot_height=int(fixture["screenshot_height"]),
    )
    assert classification.screen == ScreenName.PERMISSION_POPUP


def test_classifier_falls_back_to_unknown_below_threshold() -> None:
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        visible_text=["Lunch tomorrow?"],
        screenshot_width=1080,
        screenshot_height=2340,
        keyboard_visible=False,
        keyboard_height=0,
        visual_features={"top_band_brightness": 100.0, "bottom_band_variance": 0.0, "center_variance": 0.0},
    )
    assert classification.screen == ScreenName.UNKNOWN_SCREEN


def test_classifier_falls_back_to_unknown_on_keyboard_ambiguity() -> None:
    classifier = ScreenClassifier(selector_path=Path("data/selectors/screen_signatures.json"))
    classification = classifier.classify(
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        visible_text=["Hello"],
        screenshot_width=1080,
        screenshot_height=2340,
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
        visual_features={"top_band_brightness": 100.0, "bottom_band_variance": 0.0, "center_variance": 0.0},
    )
    assert classification.screen == ScreenName.UNKNOWN_SCREEN
    assert classification.keyboard_ambiguous is True
    assert classification.debug_info["fallback_reason"] is not None
