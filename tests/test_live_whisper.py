"""Tests for live_whisper.LiveWhisperWorker — ordering, coalescing, bounded lag."""

import threading
import time

from live_whisper import BYTES_PER_SECOND, LiveWhisperWorker


def secs(n: float, marker: int = 1) -> bytes:
    return bytes([marker, 0]) * int(n * BYTES_PER_SECOND / 2)


class Recorder:
    def __init__(self, delay=0.0, gate: threading.Event | None = None):
        self.delay = delay
        self.gate = gate
        self.calls = []  # (audio_len, interim)
        self.finals = []
        self.interims = []
        self.lock = threading.Lock()

    def transcribe(self, audio, interim):
        if self.gate is not None:
            self.gate.wait(5)
        time.sleep(self.delay)
        with self.lock:
            self.calls.append((len(audio), interim))
            n = len(self.calls)
        return f"{'i' if interim else 'f'}{n}"

    def on_final(self, text, audio):
        self.finals.append((text, len(audio)))

    def on_interim(self, text):
        self.interims.append(text)


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make(rec, **kw):
    return LiveWhisperWorker(rec.transcribe, rec.on_final, rec.on_interim, **kw)


def test_finals_submitted_while_busy_are_merged_into_one_request():
    gate = threading.Event()
    rec = Recorder(gate=gate)
    w = make(rec)
    w.submit_final(secs(1))
    assert wait_for(lambda: not w.idle)
    time.sleep(0.05)  # first request is now blocked on the gate
    w.submit_final(secs(1))
    w.submit_final(secs(1))
    w.submit_final(secs(1))
    gate.set()
    assert wait_for(lambda: len(rec.finals) == 2)
    w.close()
    # 1 request for the first segment + 1 merged request for the other 3,
    # never 4 queued requests.
    assert len(rec.calls) == 2
    assert rec.calls[1][0] > 3 * BYTES_PER_SECOND  # 3 s + merge gaps


def test_results_come_back_in_order():
    rec = Recorder(delay=0.02)
    w = make(rec)
    for _ in range(5):
        w.submit_final(secs(0.5))
        time.sleep(0.05)
    assert wait_for(lambda: w.idle and rec.finals)
    w.close()
    numbers = [int(t[1:]) for t, _ in rec.finals]
    assert numbers == sorted(numbers)


def test_backlog_is_capped_to_stay_live():
    gate = threading.Event()
    rec = Recorder(gate=gate)
    w = make(rec, max_backlog_s=5)
    w.submit_final(secs(1))
    assert wait_for(lambda: not w.idle)
    time.sleep(0.05)
    for _ in range(20):  # 20 s of speech piles up behind a stalled server
        w.submit_final(secs(1))
    gate.set()
    assert wait_for(lambda: len(rec.finals) == 2)
    w.close()
    assert rec.calls[1][0] <= 5 * BYTES_PER_SECOND


def test_interim_only_accepted_when_idle():
    gate = threading.Event()
    rec = Recorder(gate=gate)
    w = make(rec)
    w.submit_final(secs(1))
    assert wait_for(lambda: not w.idle)
    assert w.submit_interim(secs(1)) is False
    gate.set()
    assert wait_for(lambda: w.idle)
    assert w.submit_interim(secs(1)) is True
    assert wait_for(lambda: rec.interims)
    w.close()


def test_stale_interim_result_is_discarded_after_a_final():
    gate = threading.Event()
    rec = Recorder(gate=gate)
    w = make(rec)
    assert w.submit_interim(secs(1))
    assert wait_for(lambda: not w.idle)
    time.sleep(0.05)
    w.submit_final(secs(1))  # the utterance finished while the interim ran
    gate.set()
    assert wait_for(lambda: len(rec.finals) == 1)
    w.close()
    assert rec.interims == []


def test_close_finishes_pending_final():
    rec = Recorder(delay=0.05)
    w = make(rec)
    w.submit_final(secs(1))
    w.close(timeout=2)
    assert len(rec.finals) == 1


def test_transcribe_exception_does_not_kill_worker():
    calls = []

    def flaky(audio, interim):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("server down")
        return "ok"

    finals = []
    w = LiveWhisperWorker(flaky, lambda t, a: finals.append(t))
    w.submit_final(secs(0.5))
    assert wait_for(lambda: len(calls) == 1 and w.idle)
    w.submit_final(secs(0.5))
    assert wait_for(lambda: finals == ["ok"])
    w.close()
