# 🚀 QUICK START GUIDE

## What's New

- **Argos Translate** – offline translation (no internet needed)
- **Whisper API** – use a Whisper server for transcription/translation
- **Multiple translation backends** – AI, LibreTranslate, internal, whisper_translate
- **Pop‑out display** – perfect for OBS overlays
- **Session isolation** – each browser tab is independent

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
python download_vosk_models.py en-us-small # 40 MB English model
```

### 3. (Optional) Download Argos models for offline translation

```bash
python download_argos_model.py en es # English → Spanish
python download_argos_model.py --common # common language pairs
```

### 4. Start the app

```bash
python app.py
```

Open your browser to `http://localhost:7860`.

### First‑Time UI Setup

1. **Choose Recognition Engine** (Vosk recommended for offline)

2. **Select a Vosk model** from the dropdown

3. **Pick your microphone** (hardware mode)

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

3. Set width/height (e.g., 1920×200).

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
