#!/bin/sh
# First start: download a default Vosk model when vosk_models/ has none yet, so speech
# recognition works without a manual step. VOSK_DEFAULT_MODEL picks it (empty: skip); a failed
# download (no internet) is only a warning, the app still starts (Moonshine and Whisper fetch
# their own models).
if [ -n "${VOSK_DEFAULT_MODEL:-}" ]; then
    python download_vosk_models.py --if-empty "$VOSK_DEFAULT_MODEL" ||
        echo "voice-translator: could not download $VOSK_DEFAULT_MODEL; place a model in vosk_models/ (see download_vosk_models.py --list)" >&2
fi
exec "$@"
