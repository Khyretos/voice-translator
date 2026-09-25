'use strict';
/* Audio helpers for bridge.js (kept separate so they can be tested). */

// ── 48 kHz stereo → 16 kHz mono ─────────────────────────────────────────────
// 48 000 / 16 000 is exactly 3, so each output sample is the average of three
// input frames (both channels) — a boxcar low-pass + decimation in one step.
class Downsampler {
  constructor() {
    this.carry = Buffer.alloc(0);
  }
  push(chunk) {
    const buf = this.carry.length ? Buffer.concat([this.carry, chunk]) : chunk;
    const frameBytes = 4; // stereo int16
    const groups = Math.floor(buf.length / (frameBytes * 3));
    const out = Buffer.alloc(groups * 2);
    for (let g = 0; g < groups; g++) {
      const base = g * frameBytes * 3;
      let sum = 0;
      for (let f = 0; f < 3; f++) {
        sum += buf.readInt16LE(base + f * 4) + buf.readInt16LE(base + f * 4 + 2);
      }
      out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(sum / 6))), g * 2);
    }
    this.carry = Buffer.from(buf.subarray(groups * frameBytes * 3));
    return out;
  }
}

// ── Opus decoding ───────────────────────────────────────────────────────────
// One decoder per speaker, fed packet by packet. (Piping into prism's
// Decoder stream was the "voice recognition silently stops" bug: a single
// undecodable packet — which Discord's end-to-end encryption produces while
// its keys change, e.g. when the bot is moved or someone joins — errored and
// destroyed that stream, so that speaker was never heard again until a
// restart.) A bad packet is now just skipped.
function createOpusDecoder() {
  try {
    const { OpusEncoder } = require('@discordjs/opus'); // native, fastest
    const enc = new OpusEncoder(48000, 2);
    return { decode: packet => enc.decode(packet) };
  } catch (e) {
    const OpusScript = require('opusscript'); // pure JS fallback
    const dec = new OpusScript(48000, 2, OpusScript.Application.AUDIO);
    return {
      decode: packet => Buffer.from(dec.decode(packet)),
      delete: () => { try { dec.delete(); } catch (err) {} },
    };
  }
}

module.exports = { Downsampler, createOpusDecoder };
