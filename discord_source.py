"""
"Discord voice channel" audio source.

Runs discord_bridge/bridge.js (Node) as a child process. The bridge logs in
as your bot, follows one Discord user into whatever voice channel they're
in, and sends every speaker's audio separately — Discord delivers one
stream per user, so the app knows exactly who said what without having to
guess from a mixed signal. See discord_bridge/bridge.js for the protocol and
DISCORD.md for setup.

Node is used for this one part because receiving voice means decrypting
Discord's end-to-end voice encryption (DAVE), which @discordjs/voice
supports and the Python Discord libraries currently don't for received audio.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import threading
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent / "discord_bridge"
BRIDGE_SCRIPT = BRIDGE_DIR / "bridge.js"

FRAME_JSON = 1
FRAME_AUDIO = 2
_HEADER = struct.Struct(">BI")


def parse_id_list(text: str) -> list[str]:
    """User ids from free text: commas, spaces, newlines, ';' all separate."""
    return [t for t in re.split(r"[\s,;]+", text or "") if t.isdigit()]


def bridge_available() -> str | None:
    """None if the bridge can run, else a human-readable reason it can't."""
    if not shutil.which("node"):
        return "Node.js is not installed (rebuild the Docker image — see DISCORD.md)"
    if not (BRIDGE_DIR / "node_modules").exists():
        return (
            "The Discord bridge's packages aren't installed — run `npm ci` in "
            "discord_bridge/ or rebuild the Docker image (see DISCORD.md)"
        )
    return None


def read_frames(stream):
    """Yield (type, payload) frames from the bridge's stdout until EOF."""
    while True:
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            return
        ftype, length = _HEADER.unpack(header)
        payload = stream.read(length)
        if len(payload) < length:
            return
        yield ftype, payload


def split_audio(payload: bytes) -> tuple[str, bytes]:
    """Audio frame payload -> (user id, 16 kHz mono int16 PCM)."""
    (user_id,) = struct.unpack(">Q", payload[:8])
    return str(user_id), payload[8:]


class DiscordBridge:
    """
    One running bridge process. `on_event(dict)` and `on_audio(user_id, pcm)`
    are called from the reader thread; `on_exit(returncode)` once when the
    process ends (for any reason).
    """

    def __init__(
        self,
        token: str,
        follow_user_id: str,
        ignore_ids: list[str],
        on_event,
        on_audio,
        on_exit=None,
        logger=None,
        mode: str = "follow",
        fake_wav: str | None = None,
    ):
        self._on_event = on_event
        self._on_audio = on_audio
        self._on_exit = on_exit
        self._logger = logger
        env = dict(os.environ)
        env.update(
            {
                "DISCORD_TOKEN": token,
                "FOLLOW_USER_ID": follow_user_id,
                "IGNORE_USER_IDS": ",".join(ignore_ids),
                "BRIDGE_MODE": mode,
            }
        )
        if fake_wav:
            env["BRIDGE_FAKE_WAV"] = fake_wav
        self._proc = subprocess.Popen(
            ["node", str(BRIDGE_SCRIPT)],
            cwd=str(BRIDGE_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stdin_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _log(self, msg, level="info"):
        if self._logger:
            self._logger.log(msg, level=level)

    def _read_stdout(self):
        try:
            for ftype, payload in read_frames(self._proc.stdout):
                try:
                    if ftype == FRAME_AUDIO:
                        self._on_audio(*split_audio(payload))
                    elif ftype == FRAME_JSON:
                        self._on_event(json.loads(payload.decode("utf-8")))
                except Exception as exc:
                    self._log(f"Discord bridge message error: {exc}", level="error")
        finally:
            code = self._proc.wait()
            if self._on_exit:
                try:
                    self._on_exit(code)
                except Exception:
                    pass

    def _read_stderr(self):
        for raw in iter(self._proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._log(line.replace("[discord-bridge] ", "Discord: "), level="debug")

    def _send(self, command: dict):
        with self._stdin_lock:
            try:
                self._proc.stdin.write((json.dumps(command) + "\n").encode())
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def set_ignore(self, ids: list[str]):
        self._send({"type": "ignore", "ids": ids})

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def stop(self, timeout: float = 5.0):
        if self._proc.poll() is None:
            self._send({"type": "shutdown"})
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=2)


def check_bot(token: str, follow_user_id: str, timeout: float = 25.0) -> dict:
    """
    Log in once, report the bot's name, its servers and whether the followed
    user is in a voice channel right now, then exit. Used by the UI's
    "Check bot" button.
    """
    done = threading.Event()
    result: dict = {}

    def on_event(ev):
        if ev.get("type") in ("check", "error"):
            result.update(ev)
            done.set()

    def on_exit(code):
        done.set()

    bridge = DiscordBridge(
        token, follow_user_id, [], on_event, lambda *a: None, on_exit=on_exit, mode="check"
    )
    if not done.wait(timeout):
        result = {"type": "error", "message": "Timed out talking to Discord"}
    bridge.stop(timeout=3)
    return result


def invite_url(bot_id: str) -> str:
    # View Channel (1024) + Connect (1048576) + Speak (2097152)
    return (
        f"https://discord.com/oauth2/authorize?client_id={bot_id}"
        f"&scope=bot&permissions=3146752"
    )
