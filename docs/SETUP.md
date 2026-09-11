# Setup

Run commands from the repository root. Use the recording demo first; it has no device integration.

## Python environment

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev]"
python -m apps.demo.main
```

Demo: <http://127.0.0.1:8766>. If the port is occupied:

```sh
python -m uvicorn apps.demo.main:app --host 127.0.0.1 --port 8767
```

## Full controller

Copy `.env.example` to `.env` and review the settings. Keep automatic sending disabled while configuring and testing.

```sh
python -m uvicorn apps.controller.main:app --host 127.0.0.1 --port 8765
```

The controller initializes local files under `data/`. Git ignores them. The live phone view is `/`, training is `/training/simple`, the conversation simulator is `/catbot`, and diagnostics start at `/mission-control` and `/ai-core`.

The full conversation simulator requires a configured model for open-ended generation. The separate recording demo remains deterministic and needs no provider.

## Android

1. Install Android Platform Tools and put `adb` on PATH.
2. Enable USB debugging on an Android test device and authorize the computer on that device.
3. Check `adb devices`. Set `PHONE_COPILOT_ADB_DEVICE_SERIAL` when more than one device is connected.
4. Install Tesseract OCR and make `tesseract` available on PATH for screen text recognition.
5. Use a test conversation with consenting participants, keep the phone unlocked, and start in review mode.

Screen classification includes Google Messages and selected other messaging surfaces. This is not a compatibility guarantee. Recheck the detected screen and action coordinates on your device before approving an action.

`apps/android_helper/` is an optional Android notification helper. Open that folder in Android Studio; its Gradle settings define the Android build. Grant notification access only on a test device after reviewing the service.

## Owner configuration

The default identity in `libs/drafting/identity_pack.py` is a fictional example. On first startup it is written to `data/identity/identity_pack.json`. Replace it with approved facts before live drafting. Set `PHONE_COPILOT_AI_REPLY_TRAINING_OWNER_ALIASES` to the sender name used in your own local exports.

Contact profiles live at `data/contact_profiles.json`. Start with `relationship_type: "unknown"`, `auto_send_allowed: false` and `auto_draft_allowed: false`. Automation targets default to an empty allowlist. The Python models in `libs/drafting/contact_profiles.py` and `automation_targets.py` define the configuration schema.

## Ollama and external providers

Install Ollama separately and pull a model you can run locally, for example `ollama pull qwen3:4b`. Set `PHONE_COPILOT_AI_REPLY_ENABLED=true` in `.env`, keep the provider fields on `ollama`, and keep `PHONE_COPILOT_EXTERNAL_API_ENABLED=false`.

Gemini, Groq, OpenRouter and Hugging Face adapters are included. They require your own credentials and explicit external-API enablement. Enabling one can send message context to that provider; the privacy guard does not make cloud processing local. Provider model availability and pricing change independently of this repository.

## Optional vector retrieval

```sh
python -m pip install -e ".[vector]"
python scripts/build_reply_vector_db.py
```

The default backend is lexical retrieval. To use Chroma, create `data/retrieval_config.json` with `{"backend":"vector_chroma","fallback_backend":"lexical","limit":5}`. Building embeddings may download the configured model. Raw exports, embedding data and generated reports remain local.

## WhatsApp Web extension

Start the full controller, open the Chromium extensions page, enable developer mode and load `apps/browser_extension/whatsapp_web/` as an unpacked extension. Refresh WhatsApp Web afterward.

The extension requests `storage` and `debugger` permissions, plus access to WhatsApp Web and the loopback controller. It defaults to review mode and an empty archived-contact allowlist. Use a separate browser profile and test account for recordings. The extension can insert or send messages in live mode, so the isolated demo is the recommended recording surface.
