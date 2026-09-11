from pydantic import BaseModel


class DeviceInfo(BaseModel):
    serial: str
    model: str
    android_version: str
    product_name: str


class ForegroundApp(BaseModel):
    package_name: str
    activity_name: str


class KeyboardState(BaseModel):
    visible: bool | None = None
    height: int | None = None
    source: str = "unknown"
    ambiguous: bool = False


class SafeTapContext(BaseModel):
    screen: str
    before_screenshot_path: str | None = None
    after_screenshot_path: str | None = None
    expected_region_left: int | None = None
    expected_region_top: int | None = None
    expected_region_right: int | None = None
    expected_region_bottom: int | None = None
