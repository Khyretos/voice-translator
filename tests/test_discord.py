"""Tests for the Discord source: bridge framing, id parsing, and (when Node
and the bridge's packages are installed) the real bridge process in its
fake-speaker mode."""

import io
import shutil
import struct
import threading
import wave

import numpy as np
import pytest

from discord_source import (
    BRIDGE_DIR,
    DiscordBridge,
    invite_url,
    parse_id_list,
    read_frames,
    split_audio,
)


def frame(ftype: int, payload: bytes) -> bytes:
    return struct.pack(">BI", ftype, len(payload)) + payload


def test_parse_id_list_accepts_any_separator_and_skips_junk():
    assert parse_id_list("123, 456\n789;abc  1011") == ["123", "456", "789", "1011"]
    assert parse_id_list("") == []


def test_read_frames_and_split_audio():
    pcm = np.arange(10, dtype=np.int16).tobytes()
    data = frame(1, b'{"type":"ready"}') + frame(2, struct.pack(">Q", 987654321012345678) + pcm)
    frames = list(read_frames(io.BytesIO(data)))
    assert frames[0] == (1, b'{"type":"ready"}')
    uid, audio = split_audio(frames[1][1])
    assert uid == "987654321012345678" and audio == pcm


def test_read_frames_stops_on_truncated_frame():
    data = frame(1, b"{}") + struct.pack(">BI", 1, 100) + b"short"
    assert list(read_frames(io.BytesIO(data))) == [(1, b"{}")]


def test_invite_url_has_voice_permissions():
    url = invite_url("42")
    assert "client_id=42" in url and "permissions=3146752" in url


@pytest.mark.skipif(
    not shutil.which("node") or not (BRIDGE_DIR / "node_modules").exists(),
    reason="Node / discord_bridge packages not installed",
)
def test_bridge_fake_mode_streams_two_speakers(tmp_path):
    # 1 s of a 200 Hz tone at 48 kHz
    sr = 48000
    t = np.arange(sr) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    path = tmp_path / "fake.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    events, audio = [], {}
    got_audio = threading.Event()

    def on_audio(uid, data):
        audio.setdefault(uid, 0)
        audio[uid] += len(data)
        if audio[uid] > 16000:  # > 0.5 s at 16 kHz int16
            got_audio.set()

    bridge = DiscordBridge(
        "", "", [], events.append, on_audio, fake_wav=str(path)
    )
    try:
        assert got_audio.wait(10)
    finally:
        bridge.stop()
    kinds = [(e.get("type"), e.get("state")) for e in events]
    assert ("voice", "joined") in kinds
    assert {e.get("name") for e in events if e.get("type") == "speaker"} >= {"Alice", "Bob"}
    # 48 kHz stereo in, 16 kHz mono out: 20 ms packets become 640-byte chunks.
    assert all(v % 640 == 0 for v in audio.values())
    assert not bridge.alive
