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
# The bridge sends a heartbeat every 10 s. None for this long = it's frozen
# or gone, so it's restarted (without stopping the session).
_HEARTBEAT_TIMEOUT_S = 45.0
_RESTART_MIN_INTERVAL_S = 60.0
_STATUS_INTERVAL_S = 300.0  # periodic "is it still working" line in the log


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
        self._fake_wav: str | None = None
        self._gen = 0  # bumped per bridge process; events from old ones are ignored
        self._last_heartbeat = time.monotonic()
        self._last_restart = 0.0
        self._restarting = False
        self._last_status = time.monotonic()
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict:
        return {
            "speech_s": 0.0, "speakers": set(), "decoded": 0,
            "decrypt_failures": 0, "refreshes": 0, "connected": None, "channel": None,
        }

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
        self._fake_wav = fake_wav
        self._spawn(token, follow)
        if not self._joined.wait(_JOIN_TIMEOUT_S) and not self._start_error:
            self._start_error = "timed out connecting to Discord"
        if self._start_error:
            return False, f"❌ Discord: {self._start_error}"
        return True, f"✅ Listening to Discord: {self.channel_desc}"

    def _spawn(self, token: str, follow: str):
        self._gen += 1
        gen = self._gen
        self._joined.clear()
        self._start_error = None
        self._last_heartbeat = time.monotonic()
        self.bridge = DiscordBridge(
            token,
            follow,
            parse_id_list(self.app.settings.get("discord_ignore_ids", "")),
            on_event=lambda ev: self._on_event(ev) if gen == self._gen else None,
            on_audio=lambda uid, pcm: self._on_audio(uid, pcm) if gen == self._gen else None,
            on_exit=lambda code: self._on_exit(code) if gen == self._gen else None,
            logger=self.app.logger,
            fake_wav=self._fake_wav,
        )

    def restart_bridge(self, why: str):
        """Replace a frozen/crashed bridge without stopping the session."""
        now = time.monotonic()
        if self._stopping or self._restarting or now - self._last_restart < _RESTART_MIN_INTERVAL_S:
            return
        self._restarting = True
        self._last_restart = now
        threading.Thread(target=self._do_restart, args=(why,), daemon=True).start()

    def _do_restart(self, why: str):
        log = self.app.logger.log
        try:
            log(f"Discord: {why} — restarting the Discord connection", level="warning")
            old = self.bridge
            s = self.app.settings
            token = (s.get("discord_bot_token") or "").strip() or _env_token()
            follow = (s.get("discord_user_id") or "").strip()
            self._spawn(token, follow)  # new generation: the old one's exit is ignored
            if old is not None:
                old.stop(timeout=3)
            if not self._joined.wait(_JOIN_TIMEOUT_S) or self._start_error:
                self._stop_session(
                    f"🔌 Discord: couldn't reconnect ({self._start_error or 'timed out'}) "
                    f"— session stopped"
                )
            else:
                log(f"Discord: reconnected to {self.channel_desc}", level="success")
        finally:
            self._restarting = False

    def check_health(self):
        """Called about once a second by the app's watchdog."""
        if self._stopping or not self._joined.is_set():
            return
        now = time.monotonic()
        if self.bridge is not None and not self.bridge.alive:
            self.restart_bridge("the Discord bridge exited unexpectedly")
        elif now - self._last_heartbeat > _HEARTBEAT_TIMEOUT_S:
            self.restart_bridge("the Discord bridge stopped responding")
        if now - self._last_status >= _STATUS_INTERVAL_S:
            self._last_status = now
            st, self._stats = self._stats, self._new_stats()
            where = f"#{st['channel']}" if st["channel"] else self.channel_desc
            state = "connected" if st["connected"] is not False else "NOT connected"
            self.app.logger.log(
                f"Discord status: {state} to {where} — last 5 min: "
                f"{st['speech_s']:.0f}s of speech from {len(st['speakers'])} speaker(s), "
                f"{st['decoded']} audio frames decoded, {st['decrypt_failures']} decrypt "
                f"failures, {st['refreshes']} connection refreshes",
                level="info",
            )

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
        if kind == "heartbeat":
            self._last_heartbeat = time.monotonic()
            st = self._stats
            st["decoded"] += int(ev.get("decoded") or 0)
            st["decrypt_failures"] += int(ev.get("decrypt_failures") or 0)
            st["refreshes"] += int(ev.get("refreshes") or 0)
            st["connected"] = bool(ev.get("connected"))
            st["channel"] = ev.get("channel") or st["channel"]
            return
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
        elif kind == "log":
            log(ev.get("message", ""), level=ev.get("level", "info"))
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
        self._stats["speech_s"] += len(pcm) / 32000
        self._stats["speakers"].add(uid)
        if self.app.is_running:
            self.app.audio_queue.put(("discord", uid, pcm))

    def _on_exit(self, code):
        if not self._joined.is_set():
            self._start_error = self._start_error or "the Discord bridge exited"
            self._joined.set()
        elif self.app.is_running:
            # Crashed mid-session: reconnect rather than give up.
            self.restart_bridge(f"the Discord bridge exited (code {code})")

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
        quiet = True
        for sp in speakers:
            if not sp.flushed and now - sp.last_audio > _SPEAKER_IDLE_S:
                sp.flushed = True
                self._feed(sp, silence)
            if now - sp.last_audio <= _SPEAKER_IDLE_S:
                quiet = False
        if quiet:
            # Discord sends nothing while nobody talks; show silence on the
            # level meter instead of freezing at the last value.
            self.app.monitor_level = 0.0

    @property
    def active_speakers(self) -> int:
        return max(1, len(self.speakers))


def _env_token() -> str:
    import os

    return (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
