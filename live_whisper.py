"""
Low-latency request scheduling for a remote Whisper server.

The old pipeline fired one thread per VAD segment (up to 3 in parallel) and
dropped segments outright when all 3 were busy. On a server that's even a
little slower than real time that meant results arriving out of order,
chunks of speech silently missing, and — whenever the server stalled — a
growing pile of old audio still being transcribed long after it was said.

LiveWhisperWorker replaces that with one ordered worker per session:

- **Finals never queue up.** While a request is in flight, newly finished
  segments are *merged* into a single pending request instead of queueing
  behind it, so however slow the server is, at most one request is ever
  waiting. Results always come back in speaking order.
- **Bounded lag.** If the merged pending audio grows past `max_backlog_s`
  (the server can't keep up at all), the oldest audio is dropped so the
  captions skip ahead to what's being said now instead of drifting further
  and further behind.
- **Interim results only use idle time.** A partial-utterance request is
  only accepted when nothing else is waiting, only the newest one is kept,
  and its result is discarded if a final has been submitted since — so
  interims can never delay a final or show stale text.
"""

import threading
import time

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # int16 mono
# Short silence inserted between merged segments so words at the boundary
# don't run together for the decoder.
_MERGE_GAP = b"\x00\x00" * (SAMPLE_RATE // 5)  # 200 ms


class LiveWhisperWorker:
    def __init__(
        self,
        transcribe_fn,
        on_final,
        on_interim=None,
        logger=None,
        max_backlog_s: float = 12.0,
    ):
        """
        transcribe_fn(audio: bytes, interim: bool) -> str
        on_final(text: str, audio: bytes)
        on_interim(text: str)
        """
        self._transcribe = transcribe_fn
        self._on_final = on_final
        self._on_interim = on_interim
        self._logger = logger
        self._max_backlog_bytes = int(max(1.0, max_backlog_s) * BYTES_PER_SECOND)

        self._cond = threading.Condition()
        self._pending_final: bytearray | None = None
        self._pending_interim: tuple[bytes, int] | None = None
        self._in_flight = False
        self._final_seq = 0  # bumped on every submit_final
        self._closing = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _log(self, msg, level="info"):
        if self._logger:
            self._logger.log(msg, level=level)

    # ── producer side (audio thread) ─────────────────────────────────────────
    def submit_final(self, audio: bytes):
        if not audio:
            return
        with self._cond:
            if self._closing:
                return
            if self._pending_final is None:
                self._pending_final = bytearray(audio)
            else:
                self._pending_final += _MERGE_GAP + audio
            overflow = len(self._pending_final) - self._max_backlog_bytes
            if overflow > 0:
                overflow += overflow % 2  # stay sample-aligned
                del self._pending_final[:overflow]
                self._log(
                    f"Whisper server is slower than real time — skipped "
                    f"{overflow / BYTES_PER_SECOND:.1f}s of old audio to stay live",
                    level="warning",
                )
            self._final_seq += 1
            self._pending_interim = None  # superseded
            self._cond.notify()

    def submit_interim(self, audio: bytes) -> bool:
        """Queue a partial-utterance request if the worker is idle. Returns True if accepted."""
        if not audio or self._on_interim is None:
            return False
        with self._cond:
            if self._closing or self._in_flight or self._pending_final is not None:
                return False
            self._pending_interim = (bytes(audio), self._final_seq)
            self._cond.notify()
            return True

    @property
    def idle(self) -> bool:
        with self._cond:
            return (
                not self._in_flight
                and self._pending_final is None
                and self._pending_interim is None
            )

    # ── worker ───────────────────────────────────────────────────────────────
    def _run(self):
        while True:
            with self._cond:
                while (
                    self._pending_final is None
                    and self._pending_interim is None
                    and not self._closing
                ):
                    self._cond.wait()
                if self._pending_final is not None:
                    job, interim_seq = bytes(self._pending_final), None
                    self._pending_final = None
                elif self._pending_interim is not None and not self._closing:
                    job, interim_seq = self._pending_interim
                    self._pending_interim = None
                else:
                    return  # closing and nothing left to finish
                self._in_flight = True

            is_interim = interim_seq is not None
            t0 = time.monotonic()
            try:
                text = self._transcribe(job, is_interim) or ""
            except Exception as exc:
                self._log(f"Transcription error: {exc}", level="error")
                text = ""
            elapsed = time.monotonic() - t0

            with self._cond:
                self._in_flight = False
                stale = is_interim and interim_seq != self._final_seq
                self._cond.notify_all()

            audio_s = len(job) / BYTES_PER_SECOND
            self._log(
                f"Whisper {'interim' if is_interim else 'final'}: "
                f"{audio_s:.1f}s audio in {elapsed:.2f}s",
                level="debug",
            )
            if not text:
                continue
            try:
                if is_interim:
                    if not stale:
                        self._on_interim(text)
                else:
                    self._on_final(text, job)
            except Exception as exc:
                self._log(f"Result handling error: {exc}", level="error")

    def close(self, timeout: float = 10.0):
        """Finish any pending final (bounded by `timeout`), then stop the worker."""
        with self._cond:
            self._closing = True
            self._pending_interim = None
            self._cond.notify_all()
        self._thread.join(timeout=timeout)
