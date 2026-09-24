"""
Per-speaker recognition for the Discord audio source.

Every Discord user in the voice channel gets their own pipeline — their own
VAD and their own Whisper worker (or Vosk recognizer) — because Discord
delivers each person's audio as a separate stream. Results are tagged with
the speaker's user id and shown as one caption row per person (see
subtitles.SpeakerBoard).

Threads: audio arrives on the bridge's reader thread and is queued onto the
app's normal audio queue; the app's processing thread calls process() and
tick(). Each speaker's Whisper worker runs its requests on its own thread,
so one slow request never holds up another speaker.
"""

import threading
import time

import numpy as np

from discord_source import DiscordBridge, bridge_available, parse_id_list
from live_whisper import LiveWhisperWorker

# Discord stops sending packets as soon as someone stops talking, so the VAD
# would never see the silence that ends an utterance. After this long with
# no packets from a speaker, feed their VAD enough silence to close it.
_SPEAKER_IDLE_S = 0.25
_JOIN_TIMEOUT_S = 35.0


class _Speaker:
    def __init__(self, uid: str):
        self.uid = uid
        self.vad = None
        self.worker: LiveWhisperWorker | None = None
        self.recognizer = None  # Vosk
        self.last_audio = time.monotonic()
        self.last_interim = 0.0
        self.flushed = True


class DiscordPipeline:
    def __init__(self, app):
        self.app = app
        self.bridge: DiscordBridge | None = None
        self.speakers: dict[str, _Speaker] = {}
        self._lock = threading.Lock()
        self._joined = threading.Event()
        self._start_error: str | None = None
        self.channel_desc = ""
        self._stopping = False
        self._last_tick = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, fake_wav: str | None = None) -> tuple[bool, str]:
        s = self.app.settings
        problem = bridge_available()
        if problem:
            return False, f"❌ Discord: {problem}"
        token = (s.get("discord_bot_token") or "").strip() or _env_token()
        follow = (s.get("discord_user_id") or "").strip()
        if not token and not fake_wav:
            return False, "❌ Discord: set the bot token first (Audio → Discord)"
        if not follow.isdigit() and not fake_wav:
            return False, "❌ Discord: enter your Discord user ID (numbers only)"
        self.bridge = DiscordBridge(
            token,
            follow,
            parse_id_list(s.get("discord_ignore_ids", "")),
            on_event=self._on_event,
            on_audio=self._on_audio,
            on_exit=self._on_exit,
            logger=self.app.logger,
            fake_wav=fake_wav,
        )
        if not self._joined.wait(_JOIN_TIMEOUT_S) and not self._start_error:
            self._start_error = "timed out connecting to Discord"
        if self._start_error:
            return False, f"❌ Discord: {self._start_error}"
        return True, f"✅ Listening to Discord: {self.channel_desc}"

    def stop(self):
        self._stopping = True
        if self.bridge:
            self.bridge.stop()
            self.bridge = None
        with self._lock:
            speakers = list(self.speakers.values())
            self.speakers.clear()
        for sp in speakers:
            if sp.vad is not None and sp.worker is not None:
                leftover = sp.vad.flush()
                if leftover:
                    sp.worker.submit_final(leftover)
            if sp.worker is not None:
                threading.Thread(target=sp.worker.close, daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.bridge is not None and self.bridge.alive

    def set_ignore(self, text: str):
        if self.bridge:
            self.bridge.set_ignore(parse_id_list(text))

    # ── bridge callbacks (bridge reader thread) ──────────────────────────────
    def _stop_session(self, reason: str):
        if self._stopping:
            return
        self._stopping = True
        # stop_recognition joins threads and stops the bridge — never run it
        # on the bridge's own reader thread.
        threading.Thread(
            target=self.app.stop_recognition, kwargs={"reason": reason}, daemon=True
        ).start()

    def _on_event(self, ev: dict):
        kind = ev.get("type")
        log = self.app.logger.log
        if kind == "ready":
            log(f"Discord: logged in as {ev.get('bot')}", level="info")
        elif kind == "voice":
            state = ev.get("state")
            if state == "joined":
                self.channel_desc = f"#{ev.get('channel')} ({ev.get('guild')})"
                log(f"Discord: joined {self.channel_desc}", level="success")
                self._joined.set()
            elif state == "not_in_voice":
                self._start_error = (
                    "you're not in a voice channel on a server the bot is in — "
                    "join one, then press Start"
                )
                self._joined.set()
            elif state == "left":
                reason = ev.get("reason") or "left the voice channel"
                if not self._joined.is_set():
                    self._start_error = reason
                    self._joined.set()
                else:
                    self._stop_session(f"🔌 Discord: {reason} — session stopped")
        elif kind == "speaker":
            self.app.speaker_board.set_info(
                str(ev.get("user_id")), ev.get("name", ""), ev.get("avatar", "")
            )
        elif kind == "error":
            msg = ev.get("message", "unknown error")
            log(f"Discord: {msg}", level="error")
            if not self._joined.is_set():
                self._start_error = msg
                self._joined.set()

    def _on_audio(self, uid: str, pcm: bytes):
        if self.app.is_running:
            self.app.audio_queue.put(("discord", uid, pcm))

    def _on_exit(self, code):
        if not self._joined.is_set():
            self._start_error = self._start_error or "the Discord bridge exited"
            self._joined.set()
        elif self.app.is_running:
            self._stop_session("🔌 Discord connection closed — session stopped")

    # ── audio (app processing thread) ────────────────────────────────────────
    def _speaker(self, uid: str) -> _Speaker:
        sp = self.speakers.get(uid)
        if sp is not None:
            return sp
        app = self.app
        sp = _Speaker(uid)
        if app.settings.get("recognition_engine") == "vosk":
            from vosk import KaldiRecognizer

            sp.recognizer = KaldiRecognizer(app.model, 16000)
        else:
            sp.vad = app.make_vad()
            rec = app.make_whisper_recognizer()
            sp.worker = LiveWhisperWorker(
                transcribe_fn=lambda audio, interim: app._whisper_transcribe(
                    rec, audio, interim
                ),
                on_final=lambda text, audio, u=uid: app._whisper_final(text, audio, u),
                on_interim=lambda text, u=uid: app._whisper_interim(text, u),
                logger=app.logger,
            )
        with self._lock:
            self.speakers[uid] = sp
        return sp

    def process(self, uid: str, pcm: bytes):
        sp = self._speaker(uid)
        sp.last_audio = time.monotonic()
        sp.flushed = False
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size:
            self.app.monitor_level = float(np.sqrt(np.mean(samples**2)))
        self._feed(sp, pcm)

    def _feed(self, sp: _Speaker, pcm: bytes):
        app = self.app
        if sp.recognizer is not None:
            app.vosk_feed(sp.recognizer, pcm, speaker=sp.uid)
            return
        for seg in sp.vad.process_chunk(pcm):
            sp.worker.submit_final(seg)
            sp.last_interim = time.monotonic()
        if app.settings.get("whisper_interim", True):
            sp.last_interim = app.maybe_submit_interim(sp.vad, sp.worker, sp.last_interim)

    def tick(self):
        """Close utterances of speakers who went quiet (no packets from Discord)."""
        now = time.monotonic()
        if now - self._last_tick < 0.05:
            return
        self._last_tick = now
        with self._lock:
            speakers = list(self.speakers.values())
        silence_ms = int(self.app.settings.get("vad_end_silence_ms", 300)) + 100
        silence = b"\x00\x00" * (16 * silence_ms)
        for sp in speakers:
            if not sp.flushed and now - sp.last_audio > _SPEAKER_IDLE_S:
                sp.flushed = True
                self._feed(sp, silence)

    @property
    def active_speakers(self) -> int:
        return max(1, len(self.speakers))


def _env_token() -> str:
    import os

    return (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
