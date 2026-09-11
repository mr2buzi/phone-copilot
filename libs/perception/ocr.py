from __future__ import annotations

from io import BytesIO

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional runtime dependency
    pytesseract = None


class OCRExtractor:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def extract_lines(self, image_bytes: bytes) -> list[str]:
        if not self.enabled or pytesseract is None:
            return []
        try:
            image = Image.open(BytesIO(image_bytes))
            raw_text = pytesseract.image_to_string(image)
        except Exception:
            return []
        return [line.strip() for line in raw_text.splitlines() if line.strip()]
