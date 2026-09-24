"""Tests for vad.FastVAD."""

import numpy as np
import pytest

from vad import FastVAD, _F_BYTES


def silence_bytes(n_frames: int) -> bytes:
    return (np.zeros(n_frames * (_F_BYTES // 2), dtype=np.int16)).tobytes()


def tone_bytes(n_frames: int, amplitude: int = 20000) -> bytes:
    """A block of frames well above any reasonable RMS threshold."""
    n_samples = n_frames * (_F_BYTES // 2)
    t = np.arange(n_samples)
    wave = (amplitude * np.sin(2 * np.pi * 440 * t / 16000)).astype(np.int16)
    return wave.tobytes()


class TestPreprocessBlock:
    def test_preserves_length(self):
        vad = FastVAD(noise_filter_threshold=0.5)
        data = silence_bytes(10)
        out = vad.preprocess_block(data)
        assert len(out) == len(data)

    def test_noop_when_filter_disabled(self):
        vad = FastVAD(noise_filter_threshold=0.0)
        data = tone_bytes(5)
        out = vad.preprocess_block(data)
        # Filter fully off — preprocess_block short-circuits and returns
        # the input unchanged.
        assert out == data


class TestProcessChunk:
    def test_silence_produces_no_segments(self):
        vad = FastVAD(threshold_db=-30.0, end_silence_ms=100)
        segments = vad.process_chunk(silence_bytes(50))
        assert segments == []

    def test_loud_tone_eventually_dispatches_a_segment(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=50)
        # Speech (350ms, comfortably over the 300ms minimum dispatch floor),
        # then enough silence frames to close the segment.
        segments = vad.process_chunk(tone_bytes(35))
        segments += vad.process_chunk(silence_bytes(20))
        assert len(segments) == 1
        # Segment bytes should be non-trivial in length.
        assert len(segments[0]) > 0

    def test_short_burst_is_dropped_as_noise(self):
        # A burst shorter than _MIN_DISPATCH_FRAMES should never produce a
        # dispatched segment, even after silence closes it.
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=50)
        segments = vad.process_chunk(tone_bytes(2))  # well under min dispatch
        segments += vad.process_chunk(silence_bytes(20))
        assert segments == []


class TestHotReload:
    def test_update_threshold_changes_gate(self):
        vad = FastVAD(threshold_db=-10.0)
        rms_before = vad._rms_floor
        vad.update_threshold(-40.0)
        assert vad._rms_floor != rms_before
        assert vad.threshold == -40.0

    def test_update_end_silence_ms(self):
        vad = FastVAD(end_silence_ms=100)
        vad.update_end_silence_ms(400)
        assert vad._end_silence_frames == 40  # 400ms / 10ms per frame

    def test_end_silence_frames_has_floor_of_2(self):
        vad = FastVAD(end_silence_ms=0)
        assert vad._end_silence_frames >= 2


class TestFlushAndReset:
    def test_flush_with_no_active_segment_returns_none(self):
        vad = FastVAD()
        assert vad.flush() is None

    def test_reset_clears_state(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=1000)
        vad.process_chunk(tone_bytes(10))  # opens a segment, doesn't close it
        assert vad._in_speech is True
        vad.reset()
        assert vad._in_speech is False
        assert vad._segment == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestMaxSegment:
    def test_continuous_speech_is_cut_into_pieces(self):
        # 5 s of uninterrupted speech with a 2 s cap should dispatch pieces
        # while still speaking, instead of nothing until the speaker pauses.
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300, max_segment_ms=2000)
        segments = []
        for _ in range(50):  # 50 × 100 ms
            segments += vad.process_chunk(tone_bytes(10))
        assert len(segments) >= 2
        assert vad.in_speech  # the tail is still an open utterance
        for seg in segments:
            assert len(seg) <= 2000 // 10 * _F_BYTES

    def test_cut_prefers_the_quietest_recent_frame(self):
        vad = FastVAD(threshold_db=-60.0, end_silence_ms=2000, max_segment_ms=2000)
        # 1.5 s loud, 50 ms quieter (still "speech"), then loud until the cap.
        segments = vad.process_chunk(tone_bytes(150))
        segments += vad.process_chunk(tone_bytes(5, amplitude=600))
        segments += vad.process_chunk(tone_bytes(60))
        assert len(segments) == 1
        # The cut lands inside the quiet stretch: head ≈ 1.5 s + preroll.
        head_frames = len(segments[0]) // _F_BYTES
        assert 150 <= head_frames <= 156

    def test_zero_means_unlimited(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300, max_segment_ms=0)
        segments = []
        for _ in range(50):
            segments += vad.process_chunk(tone_bytes(10))
        assert segments == []

    def test_current_segment_exposes_open_utterance(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300)
        assert vad.current_segment() == b""
        vad.process_chunk(tone_bytes(50))
        assert vad.in_speech
        assert vad.current_segment_ms() >= 500
        assert len(vad.current_segment()) == vad.current_segment_ms() // 10 * _F_BYTES


class TestTransientRejection:
    """Desk knocks / clicks must never reach Whisper (they're what it
    hallucinates "Thank you." / "Subscribe" on)."""

    def test_desk_knock_followed_by_silence_is_not_dispatched(self):
        # 30 ms loud burst + a long silence. The old check counted the
        # silence wait as part of the "utterance" and sent this to Whisper.
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300)
        segments = vad.process_chunk(tone_bytes(3, amplitude=30000))
        segments += vad.process_chunk(silence_bytes(60))
        assert segments == []

    def test_short_burst_below_min_speech_is_rejected_and_counted(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300, min_speech_ms=200)
        segments = vad.process_chunk(tone_bytes(12))  # 120 ms
        segments += vad.process_chunk(silence_bytes(60))
        assert segments == []
        assert vad.rejected == 1

    def test_real_utterance_passes_with_trailing_silence_trimmed(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=300, min_speech_ms=200)
        segments = vad.process_chunk(tone_bytes(40))  # 400 ms "speech"
        segments += vad.process_chunk(silence_bytes(60))
        assert len(segments) == 1
        frames = len(segments[0]) // _F_BYTES
        # speech + ≤ preroll + only ~100 ms of the 300 ms silence wait
        assert 40 <= frames <= 40 + 6 + 10

    def test_single_loud_frame_cannot_open_a_segment(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=50)
        vad.process_chunk(tone_bytes(1) + silence_bytes(1) + tone_bytes(1))
        assert not vad.in_speech
