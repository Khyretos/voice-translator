"""
Opening a server-side input device at whatever sample rate it supports.

Recognition works on 16 kHz mono int16, but many devices (especially raw
ALSA hardware inside Docker, where PulseAudio isn't resampling for us) can't
record at 16 kHz and PortAudio refuses with "Invalid sample rate
[PaErrorCode -9997]". Instead of failing, open the device at its own rate
and resample to 16 kHz in the callback, so the rest of the app always sees
16 kHz mono int16 in ~30 ms blocks, exactly as before.
"""

import numpy as np
import sounddevice as sd

TARGET_RATE = 16000
BLOCK_MS = 30


class StreamResampler:
    """
    Streaming linear-interpolation resampler (float32 mono) that carries its
    fractional position and last sample across blocks, so block boundaries
    don't click. A short moving average in front limits aliasing when
    downsampling (e.g. 48 kHz → 16 kHz).
    """

    def __init__(self, in_rate: float, out_rate: int = TARGET_RATE):
        self.step = float(in_rate) / out_rate
        self.t = 0.0  # next output position, in input samples; -1 = previous block's last
        self.prev = 0.0
        self.taps = max(1, min(8, int(round(self.step))))
        self._hist = np.zeros(self.taps - 1, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.taps > 1:
            padded = np.concatenate((self._hist, x))
            self._hist = padded[-(self.taps - 1):].copy()
            x = np.convolve(padded, np.full(self.taps, 1.0 / self.taps, np.float32), "valid")
        L = x.size
        if self.t >= L - 1:
            n = 0
        else:
            n = int(np.ceil((L - 1 - self.t) / self.step))
        pos = self.t + np.arange(n) * self.step
        xs = np.concatenate(([self.prev], x))  # xs[i + 1] == x[i]
        out = np.interp(pos + 1.0, np.arange(L + 1), xs).astype(np.float32)
        self.t = self.t + n * self.step - L
        self.prev = float(x[-1])
        return out


def _candidate_rates(device) -> list[float]:
    rates = [TARGET_RATE]
    try:
        default = float(sd.query_devices(device, "input")["default_samplerate"])
        if default and default not in rates:
            rates.append(default)
    except Exception:
        pass
    for r in (48000, 44100, 32000, 22050, 96000):
        if r not in rates:
            rates.append(r)
    return rates


def open_input_stream(device, callback, logger=None):
    """
    Open and return (not started) a PortAudio input stream on `device` that
    calls `callback(indata_bytes, frames, time_info, status)` with 16 kHz
    mono int16 audio — the same signature sounddevice's RawInputStream
    callbacks use, so existing callbacks work unchanged.
    """
    last_error = None
    for rate in _candidate_rates(device):
        if rate == TARGET_RATE:
            try:
                sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
                return sd.RawInputStream(
                    samplerate=rate,
                    blocksize=TARGET_RATE * BLOCK_MS // 1000,
                    device=device,
                    dtype="int16",
                    channels=1,
                    callback=callback,
                )
            except Exception as exc:
                last_error = exc
                continue
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
        except Exception as exc:
            last_error = exc
            continue
        resampler = StreamResampler(rate)

        def _cb(indata, frames, time_info, status, _rs=resampler):
            mono = np.asarray(indata, dtype=np.float32).reshape(-1)
            out = _rs.process(mono)
            pcm = (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16)
            callback(pcm.tobytes(), pcm.size, time_info, status)

        stream = sd.InputStream(
            samplerate=rate,
            blocksize=int(rate * BLOCK_MS / 1000),
            device=device,
            dtype="float32",
            channels=1,
            callback=_cb,
        )
        if logger:
            logger.log(
                f"Input device doesn't support {TARGET_RATE} Hz — recording at "
                f"{int(rate)} Hz and resampling to {TARGET_RATE} Hz",
                level="info",
            )
        return stream
    raise RuntimeError(f"No usable sample rate for this input device ({last_error})")
