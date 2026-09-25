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

## Sessions & Audio Sources

### The session stopped by itself ("🔌 … session stopped")

The audio source went away: a server device stopped delivering audio
(unplugged), the browser mic was disconnected, tab sharing was stopped, the
page was closed, or (Discord) you left the voice channel. The reason is in
the Status box and the log. Silence never triggers this. Change the delay
with **Auto-stop when audio is lost** in the Audio settings (0 = never).

### Browser mic stops after a network blip or page reload

The page reconnects its audio by itself; the status shows *Connection lost —
reconnecting…* in the meantime. After a reload, a running **Browser
Microphone** session re-attaches the mic automatically. If the status says
the browser paused audio, click anywhere on the page. **Tab / System Audio**
has to be shared again: press ▶️ Start (on a running session that just
re-attaches the audio).

### Start shows "error" / console says `withWs is not defined` (or another ReferenceError)

The browser is running an old or hand-merged copy of the page's script.
Hard-reload (Ctrl+Shift+R). If it persists, the server has a mixed version
of `voice_translator.py`; replace it with the repository version (don't
merge it by hand) and rebuild. Check with
`docker exec voice-translator grep -c withWs /app/voice_translator.py`,
which should print `0`.

### Browser audio never connects (WebSocket 404 in the browser console)

uvicorn has no WebSocket support installed. `websockets` is in
`requirements.txt`; rebuild the image (or `pip install websockets`).

### `ModuleNotFoundError: No module named '…'` at startup

A `.py` file is missing on the server. Copy **all** `.py` files (and the
`discord_bridge/` folder) and rebuild with `docker compose up -d --build`;
the files are baked into the image, so restarting alone isn't enough.

### "Invalid sample rate [PaErrorCode -9997]" (Server Audio Device)

Handled automatically now: devices that can't record at 16 kHz are opened at
their own rate and resampled (the log says so). If it still fails, check the
**Input Device** dropdown: device numbers can change after a rebuild or when
devices are plugged in or removed.

### Popout URL shows "Not found" in OBS

The popout ID belongs to a session; it's saved in `settings/<session>.json`
(make sure the `./settings` volume is mounted). A custom ID is saved when
you press Enter or click away from the field; the Status box confirms it.
Each ID can belong to only one session.

## Whisper: Latency & Hallucinations

### Captions lag behind / old audio being processed

- Keep **Low-latency decoding** and **Live partial captions** on (Whisper →
  Live mode) and lower **Max segment length** for more frequent updates.
- The log shows every request as `Whisper final: 2.8s audio in 0.41s`. If
  the second number is regularly larger than the first, the Whisper server
  itself is slower than real time: use a smaller model or a GPU.
- A "skipped … old audio to stay live" warning means the server couldn't
  keep up and the app jumped ahead instead of falling further behind.

### "Thank you" / "Subscribe" / subtitle credits appear from nowhere

These are Whisper hallucinations on noise or silence. Raise **Ignore sounds
shorter than** (Audio → Voice Detection) if knocks or clicks still get
through, and raise the **Detection threshold** if background noise opens
segments. Keep **Condition on previous text** off. See
[RECOGNITION_QUALITY.md](./RECOGNITION_QUALITY.md).

## Discord

See [DISCORD.md](./DISCORD.md) for setup. Press **🔍 Check bot** first; it
reports most problems directly.

- **"you're not in a voice channel on a server the bot is in"**: join a voice
  channel first, and make sure the bot was invited to that server (Check bot
  prints an invite link).
- **"Discord login failed"**: the bot token is wrong or was reset.
- **"Node.js is not installed" / "packages aren't installed"**: rebuild the
  Docker image (it installs Node and the bridge). Outside Docker, install
  Node.js 22.12+ and run `npm ci` in `discord_bridge/`.
- **"Could not connect to #channel"**: the bot lacks the *Connect*
  permission in that channel.
- **Captions silently stop; the log shows `decode error … Invalid packet`**:
  fixed. Discord's end-to-end voice encryption briefly delivers audio that
  can't be decoded while its keys change (someone joins/leaves, the bot is
  moved). That used to shut down the speaker's decoder for good; bad packets
  are now skipped. If audio stays undecodable for several seconds, the bot
  rejoins the channel by itself and logs `could not be decrypted —
  reconnecting` as a warning.
- **Captions stop after a while with no error** (level meter frozen): the
  app now recovers from this by itself. If audio is arriving but can't be
  decrypted for 5 s, or no audio arrives for 2 minutes while people are in
  the channel, the voice connection is refreshed without leaving the channel
  (no join/leave sound). If the bridge itself freezes or crashes, it's
  restarted. Each recovery is logged as a warning. Every 5 minutes the log
  shows a status line, e.g. `Discord status: connected to #general — last 5
  min: 212s of speech from 3 speaker(s), 10400 audio frames decoded, 0
  decrypt failures, 0 connection refreshes`. If captions stop, that line
  shows where: no speech received (Discord side), speech received but
  nothing decoded (encryption), or speech received and decoded but no text
  (Whisper/translation side).
- **The bot was dragged to another channel**: it goes back to the channel
  of the user it follows (logged as a warning).
- **Someone isn't captioned**: check they're not in **Ignore these user IDs**,
  and that **Max speakers on screen** isn't lower than the number of people
  talking at once.
- For more detail, the log shows the bridge's own messages as
  `Discord: …` (debug level).
