"""Discord bridge audio helpers (discord_bridge/audio.js), run under Node."""

import json
import shutil
import subprocess

import pytest

from discord_source import BRIDGE_DIR

pytestmark = pytest.mark.skipif(
    not shutil.which("node") or not (BRIDGE_DIR / "node_modules").exists(),
    reason="Node / discord_bridge packages not installed",
)

_SCRIPT = r"""
const { createOpusDecoder, Downsampler } = require('./audio');
const { OpusEncoder } = require('@discordjs/opus');
const enc = new OpusEncoder(48000, 2);
const pcm = Buffer.alloc(960 * 4);
for (let i = 0; i < 960; i++) {
  const v = Math.round(8000 * Math.sin(i / 10));
  pcm.writeInt16LE(v, i * 4); pcm.writeInt16LE(v, i * 4 + 2);
}
const good = enc.encode(pcm);
const corrupt = Buffer.from(Array.from({ length: 120 }, (_, i) => (i * 97 + 13) & 255));
const dec = createOpusDecoder();
const ds = new Downsampler();
let errors = 0, decoded = 0, out = 0;
const packets = [good, corrupt, corrupt, ...Array(20).fill(good)];
for (const p of packets) {
  try { const frame = dec.decode(p); decoded++; out += ds.push(frame).length; }
  catch (e) { errors++; }
}
console.log(JSON.stringify({ errors, decoded, out }));
"""


def test_corrupt_packet_does_not_stop_the_speakers_audio():
    # The old piped decoder was destroyed by the first undecodable packet
    # (which Discord's E2E encryption produces while keys change), silently
    # ending that speaker's audio until a restart.
    res = subprocess.run(
        ["node", "-e", _SCRIPT], cwd=BRIDGE_DIR, capture_output=True, text=True, timeout=30
    )
    assert res.returncode == 0, res.stderr
    r = json.loads(res.stdout)
    assert r["errors"] == 2
    assert r["decoded"] == 21  # every valid packet, including all after the bad ones
    assert r["out"] == 21 * 320 * 2  # 20 ms at 16 kHz mono int16 each
