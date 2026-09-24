"""Tests for audio_input.StreamResampler (no audio device needed)."""

import numpy as np
import pytest

sd = pytest.importorskip("sounddevice")  # audio_input imports it at module level
from audio_input import StreamResampler  # noqa: E402


@pytest.mark.parametrize("rate", [48000, 44100, 32000, 22050])
def test_output_length_matches_rate_across_blocks(rate):
    rs = StreamResampler(rate)
    block = int(rate * 0.03)
    total = sum(rs.process(np.zeros(block, np.float32)).size for _ in range(100))
    expected = 100 * block * 16000 / rate
    assert abs(total - expected) <= 1


def test_tone_frequency_preserved_without_block_glitches():
    rate = 48000
    t = np.arange(rate) / rate
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rs = StreamResampler(rate)
    out = np.concatenate([rs.process(x[i:i + 1440]) for i in range(0, rate, 1440)])
    # The 3-tap moving average delays the signal by 1 input sample.
    ref = 0.5 * np.sin(2 * np.pi * 440 * (np.arange(out.size) / 16000 - 1 / rate))
    err = np.abs(out[100:] - ref[100:])
    assert err.max() < 0.01


def test_falls_back_to_native_rate_when_16k_is_rejected(monkeypatch):
    import audio_input

    def check(device=None, samplerate=None, **kw):
        if samplerate == 16000:
            raise sd.PortAudioError("Invalid sample rate [PaErrorCode -9997]")

    created = {}

    class FakeStream:
        def __init__(self, samplerate, blocksize, callback, **kw):
            created.update(rate=samplerate, blocksize=blocksize, callback=callback)

    monkeypatch.setattr(audio_input.sd, "check_input_settings", check)
    monkeypatch.setattr(
        audio_input.sd, "query_devices", lambda *a, **k: {"default_samplerate": 48000.0}
    )
    monkeypatch.setattr(audio_input.sd, "InputStream", FakeStream)

    got = []
    audio_input.open_input_stream(3, lambda data, frames, ti, st: got.append(data))
    assert created["rate"] == 48000.0
    assert created["blocksize"] == 1440  # 30 ms

    # The app's callback still receives 16 kHz mono int16 bytes.
    created["callback"](np.zeros((1440, 1), np.float32), 1440, None, None)
    assert len(got) == 1 and len(got[0]) == 480 * 2
