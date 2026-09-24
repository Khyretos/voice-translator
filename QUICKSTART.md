# 🚀 QUICK START GUIDE

## What's New

- **Argos Translate** – offline translation (no internet needed)
- **Whisper API** – use a Whisper server for transcription/translation
- **Multiple translation backends** – AI, LibreTranslate, internal, whisper_translate
- **Pop‑out display** – perfect for OBS overlays; the popout ID is saved, so the OBS URL survives restarts
- **Named sessions** – `?session=<name>` gives you a session with its own saved settings that survives reloads and reconnects
- **Discord voice channel** – caption everyone in a Discord call separately, with avatars ([DISCORD.md](./DISCORD.md))
- **Live Whisper** – captions while you talk, never piling up behind a slow server

## Fastest Way to Get Started

### 1. Run the setup script

**Linux/macOS:**

```Bash
chmod +x setup.sh
./setup.sh
```

Windows:

```text
setup.bat
```

### 2. Download a Vosk model (for offline recognition)

```bash
# Activate virtual environment (if not already)
source venv/bin/activate # or venv\Scripts\activate on Windows
python download_vosk_models.py --lang en-us --small # current small (~40 MB) English model
python download_vosk_models.py --list                # everything available
```

### 3. (Optional) Download Argos models for offline translation

```bash
python download_argos_model.py en es # English → Spanish
python download_argos_model.py --common # common language pairs
```

### 4. Start the app

```bash
python voice_translator.py
```

(or `start.bat` on Windows). With Docker instead: `docker compose up -d --build`.

Open your browser to `http://localhost:7860`.

### First‑Time UI Setup

1. **Choose Recognition Engine** (Vosk recommended for offline)

2. **Select a Vosk model** from the dropdown

3. **Pick your audio source**: a server audio device, this browser's microphone,
   a shared browser tab / system audio, or a Discord voice channel

4. **Enable Translation** and choose a mode:
   - **Argos** – offline, requires models

   - **AI** – for Ollama/OpenAI

   - **Whisper Translate** – if you have a Whisper server

   - **LibreTranslate** – self‑hosted or cloud

   - **Internal** – easiest (uses Google Translate)

5. **Set languages** (e.g., source: en-US, target: es)

6. Click **Start**

### OBS Integration

1. After starting, copy the **Popout** URL from the UI.

2. In OBS, add a **Browser Source** and paste the URL.

3. Set width/height (e.g., 1920×200 for a caption bar, or 1920×1080 with
   **Vertical position: Bottom** for subtitles over the whole stream).

4. (Optional) Add custom CSS to remove background:

   ```css
   body {
     background-color: rgba(0, 0, 0, 0);
     overflow: hidden;
   }
   ```

### Next Steps

- Read the full [README.md](./README.md) for detailed configuration.
- Check [CONFIG_EXAMPLES.txt](./CONFIG_EXAMPLES.txt) for ready‑to‑use setups.
- If something doesn't work, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
