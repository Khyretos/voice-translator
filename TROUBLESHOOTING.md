# 🔧 Troubleshooting Guide

## Argos Translate Issues

### "Argos not available" or translation returns placeholder

- Ensure `argostranslate` is installed (`pip install argostranslate`).
- Download required language pairs with `download_argos_model.py`.
- Verify that models are installed in `argos_models/` and that the directory is writable.
- Check the logs for specific errors (e.g., package index update failure).

### Argos package index update fails

- This may happen behind a restrictive firewall. You can manually download packages from [Argos](https://www.argosopentech.com/) and place them in `argos_models/`.
- The app will still use already installed models.

## Whisper API Issues

### Cannot connect to Whisper server

- Verify the server is running and accessible from the machine running the translator.
- Test with `curl`:

  ```bash
  curl http://localhost:9000/health
  ```

- If using a remote server, ensure the host and port are correct and firewalls allow the connection.

### Transcription returns empty or nonsense

- Check that the audio is reaching the server (the server logs may show requests).

- Ensure the Whisper model supports the language you specified.

- Try without specifying a language to let Whisper auto‑detect.

### Whisper Translate returns nothing

- Make sure the API endpoint supports the `/audio/translations` route (some Whisper implementations only provide transcriptions).

- The `whisper_translate` mode sends the audio and expects translated English text; if your server doesn't support that, use a different translation mode.

## General Issues

### No audio input

- Check microphone permissions in your OS.
- In hardware mode, verify the correct device is selected.
- On Linux, add your user to the `audio` group and reboot.

### Model not loading (Vosk)

- Ensure the model directory contains the correct files (e.g., `am/`, `conf/`).
- Click **Refresh Models** after placing a new model.

### Translation not working

- For **Internal** mode, an internet connection is required.
- For **AI** mode, check the API endpoint and key.
- For **LibreTranslate**, ensure the service is running and the API key (if any) is correct.
- For **Argos**, verify the language pair is installed.

### Docker audio issues

- Make sure the container has access to the host's sound devices:

  ```yaml
  devices:
    - /dev/snd:/dev/snd
  ```

- If using PulseAudio, additional configuration may be needed.

### High memory usage

- Vosk large models consume significant RAM; use the “small” variants.
- Argos models also consume memory; uninstall unused language pairs.
- Restrict Docker memory: `mem_limit: 4g` in `docker-compose.yml`.
