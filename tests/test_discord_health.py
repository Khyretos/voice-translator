"""Discord self-healing: heartbeat watch, bridge restart, status line, level meter, thread watchdog."""

import queue
import threading
import time
import types
import wave

import numpy as np
import pytest

import discord_pipeline as dp
from discord_pipeline import DiscordPipeline
from discord_source import DiscordBridge


class _Logger:
    def __init__(self):
        self.lines = []

    def log(self, message, level="info"):
        self.lines.append((level, message))


class _App:
    def __init__(self):
        self.settings = {}
        self.logger = _Logger()
        self.is_running = True
        self.monitor_level = 0.7
        self.audio_queue = queue.Queue()
        self.stopped = threading.Event()
        self.stop_reason = None

    def stop_recognition(self, reason=None):
        self.stop_reason = reason
        self.stopped.set()


class _Bridge:
    def __init__(self, alive=True):
        self.alive = alive
        self.stopped = False

    def stop(self, timeout=None):
        self.stopped = True
        self.alive = False


def _joined_pipeline(app=None):
    p = DiscordPipeline(app or _App())
    p.bridge = _Bridge()
    p._joined.set()
    p.channel_desc = "#general (Test)"
    return p


def _record_restarts(p):
    calls = []
    p.restart_bridge = calls.append
    return calls


def test_missing_heartbeat_restarts_the_bridge():
    p = _joined_pipeline()
    calls = _record_restarts(p)
    p.check_health()
    assert calls == []
    p._last_heartbeat = time.monotonic() - dp._HEARTBEAT_TIMEOUT_S - 1
    p.check_health()
    assert calls == ["the Discord bridge stopped responding"]


def test_heartbeat_event_keeps_it_alive_and_counts():
    p = _joined_pipeline()
    calls = _record_restarts(p)
    p._last_heartbeat = time.monotonic() - dp._HEARTBEAT_TIMEOUT_S - 1
    p._on_event({"type": "heartbeat", "connected": True, "channel": "general",
                 "decoded": 50, "decrypt_failures": 2, "refreshes": 1})
    p.check_health()
    assert calls == []
    assert p._stats["decoded"] == 50 and p._stats["decrypt_failures"] == 2
    assert p._stats["refreshes"] == 1 and p._stats["channel"] == "general"


def test_dead_bridge_process_restarts_it():
    p = _joined_pipeline()
    calls = _record_restarts(p)
    p.bridge.alive = False
    p.check_health()
    assert calls == ["the Discord bridge exited unexpectedly"]


def test_exit_mid_session_restarts_instead_of_stopping():
    app = _App()
    p = _joined_pipeline(app)
    calls = _record_restarts(p)
    p._on_exit(1)
    assert calls == ["the Discord bridge exited (code 1)"]
    assert not app.stopped.is_set()


def test_nothing_happens_before_joining_or_while_stopping():
    p = DiscordPipeline(_App())
    calls = _record_restarts(p)
    p._last_heartbeat = 0
    p.check_health()  # not joined yet
    p._joined.set()
    p._stopping = True
    p.check_health()
    assert calls == []


def test_status_line_every_interval():
    app = _App()
    p = _joined_pipeline(app)
    _record_restarts(p)
    p._on_event({"type": "heartbeat", "connected": True, "channel": "general", "decoded": 7})
    p._on_audio("42", b"\x00" * 32000)  # 1 s of 16 kHz int16
    p._last_status = time.monotonic() - dp._STATUS_INTERVAL_S - 1
    p.check_health()
    status = [m for _, m in app.logger.lines if m.startswith("Discord status")]
    assert len(status) == 1
    assert "connected to #general" in status[0]
    assert "1s of speech from 1 speaker(s)" in status[0] and "7 audio frames" in status[0]
    assert p._stats["decoded"] == 0  # counters start over


def test_restart_reconnects_without_stopping_the_session(monkeypatch):
    app = _App()
    p = _joined_pipeline(app)
    old = p.bridge

    def spawn(token, follow):
        p.bridge = _Bridge()
        p._joined.set()

    monkeypatch.setattr(p, "_spawn", spawn)
    p._do_restart("the Discord bridge stopped responding")
    assert old.stopped and p.bridge is not old
    assert any("reconnected" in m for _, m in app.logger.lines)
    assert not app.stopped.is_set() and not p._restarting


def test_failed_restart_stops_the_session(monkeypatch):
    app = _App()
    p = _joined_pipeline(app)
    monkeypatch.setattr(dp, "_JOIN_TIMEOUT_S", 0.1)

    def spawn(token, follow):
        p._start_error = "invalid token"
        p._joined.set()

    monkeypatch.setattr(p, "_spawn", spawn)
    p._do_restart("the Discord bridge exited (code 1)")
    assert app.stopped.wait(5)
    assert "couldn't reconnect (invalid token)" in app.stop_reason


def test_restart_is_rate_limited(monkeypatch):
    p = _joined_pipeline()
    started = []
    monkeypatch.setattr(p, "_do_restart", started.append)
    p.restart_bridge("first")
    p.restart_bridge("second")  # within _RESTART_MIN_INTERVAL_S
    deadline = time.time() + 2
    while not started and time.time() < deadline:
        time.sleep(0.01)
    assert started == ["first"]


def test_level_meter_drops_to_zero_when_nobody_talks():
    app = _App()
    app.settings["vad_end_silence_ms"] = 300
    p = _joined_pipeline(app)
    p.tick()
    assert app.monitor_level == 0.0


def test_watchdog_restarts_a_dead_processing_thread():
    vt = pytest.importorskip("voice_translator")
    ran = threading.Event()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    health = []
    stub = types.SimpleNamespace(
        is_running=True,
        process_thread=dead,
        logger=_Logger(),
        process_audio_hardware=ran.set,
        discord=types.SimpleNamespace(check_health=lambda: health.append(1)),
        settings={"audio_watchdog_seconds": 0},
    )
    vt.VoiceTranslatorApp.check_audio_watchdog(stub)
    assert ran.wait(5)
    assert stub.process_thread is not dead
    assert health == [1]
    assert any("restarting it" in m for _, m in stub.logger.lines)


def test_fake_bridge_sends_heartbeats(tmp_path):
    import shutil
    from discord_source import BRIDGE_DIR
    if not shutil.which("node") or not (BRIDGE_DIR / "node_modules").exists():
        pytest.skip("Node / discord_bridge packages not installed")
    path = tmp_path / "fake.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(48000, dtype=np.int16).tobytes())
    beat = threading.Event()
    events = []

    def on_event(ev):
        events.append(ev)
        if ev.get("type") == "heartbeat":
            beat.set()

    bridge = DiscordBridge("", "", [], on_event, lambda uid, pcm: None, fake_wav=str(path))
    try:
        assert beat.wait(15)
    finally:
        bridge.stop()
    hb = next(e for e in events if e.get("type") == "heartbeat")
    assert hb["connected"] is True
