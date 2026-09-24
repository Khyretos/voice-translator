import argparse
import asyncio
import gc
import hashlib
import html
import json
import os
import queue
import re
import secrets
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from vosk import KaldiRecognizer, Model

from logger import Logger
import translators as tmod
from translators import TranslationService

# ── Recognizer / translation backends (see recognizers.py) ───────────────────
from recognizers import (
    ARGOS_AVAILABLE,
    MOONSHINE_LANGUAGES,
    ArgosTranslator,
    MoonshineRecognizer,
    WhisperRecognizer,
    _MOONSHINE_AVAILABLE,
    dots_or_stars,
    is_whisper_hallucination,
)
from vad import FastVAD, _WRTCVAD_AVAILABLE
from live_whisper import LiveWhisperWorker
from audio_input import open_input_stream
from subtitles import SubtitleManager
from session import (
    SessionSlugMiddleware,
    get_slug,
    register_slug,
)
from settings_store import (
    SETTINGS_DIR,
    find_slug_by_popout_id,
    load_saved_settings,
    persist_settings,
)

if ARGOS_AVAILABLE:
    # Needed directly (not just via ArgosTranslator) for the Argos settings
    # panel in create_ui(), which lists installed language pairs.
    import argostranslate.translate


# ── Session storage ───────────────────────────────────────────────────────────
# Keyed by *slug* (a persistent identity — see SessionSlugMiddleware below),
# not by Gradio's ephemeral per-page-load session_hash. This is what lets a
# tab reload / reconnect / a second tab re-attach to the same running
# session instead of spawning a brand new one.
SESSION_APPS: dict = {}
SESSION_LOCK = threading.Lock()

# Session slug resolution (SessionSlugMiddleware, get_slug, sanitize_slug)
# and per-slug settings persistence (load_saved_settings, persist_settings,
# PERSISTABLE_KEYS, SETTINGS_DIR) now live in session.py / settings_store.py
# — imported above. See SESSIONS.md.

# ── Live-pipeline tuning ──────────────────────────────────────────────────────
# Whisper interim captions: at most one partial request per interval, and
# only once the utterance has enough audio for a meaningful guess.
_INTERIM_INTERVAL_S = 0.8
_INTERIM_MIN_AUDIO_MS = 800
# If audio processing falls this far behind (blocks are 30–40 ms), drop the
# oldest blocks instead of transcribing ever-older audio.
_AUDIO_BACKLOG_MAX_BLOCKS = 100  # ~3–4 s
_AUDIO_BACKLOG_KEEP_BLOCKS = 10
# Audio watchdog (see VoiceTranslatorApp.check_audio_watchdog)
_HW_WATCHDOG_MAX_S = 3.0  # a server device calls back every 30 ms, even in silence
_BROWSER_START_GRACE_S = 30.0  # time for the mic prompt / screen-share picker

# ── Font helpers ──────────────────────────────────────────────────────────────
def get_available_fonts():
    """Scan the fonts/ directory for supported font files."""
    fonts_dir = Path("fonts")
    fonts_dir.mkdir(exist_ok=True)
    fonts = []
    for ext in (".ttf", ".otf", ".woff", ".woff2"):
        for file in fonts_dir.glob(f"*{ext}"):
            fonts.append((f"[Custom] {file.stem}", file.name))
    return fonts


SYSTEM_FONTS = [
    ("Arial", "Arial"),
    ("Helvetica", "Helvetica"),
    ("Courier New", "Courier New"),
    ("Georgia", "Georgia"),
    ("Verdana", "Verdana"),
    ("Times New Roman", "Times New Roman"),
    ("Comic Sans MS", "Comic Sans MS"),
    ("Impact", "Impact"),
]

# ── Browser-side JavaScript ───────────────────────────────────────────────────
js = r"""
<script>
// ─────────────────────────────────────────────────────────────────────────────
// Browser side of the app. Everything here talks to the server with plain
// fetch()/WebSocket calls, never through Gradio's event queue, so a slow or
// briefly-dropped Gradio connection can't stall the display or cut the audio.
//
//  • Audio capture runs in an AudioWorklet (its own audio thread), resampled
//    to 16 kHz there and sent in 40 ms chunks. The old ScriptProcessorNode ran
//    on the page's main thread and dropped audio whenever the page was busy.
//  • The audio WebSocket reconnects by itself after a network blip.
//  • If the mic is unplugged / the tab share is stopped, the session is
//    stopped on the server immediately (see vtSourceLost). The server also
//    has its own watchdog for when this page can't tell it (tab closed).
//  • After a page reload, a running browser-mic session re-attaches the mic
//    automatically.
// ─────────────────────────────────────────────────────────────────────────────
window.__audioStreamActive = false;
window.__micTestActive = false;
window.__hwLevelInterval = null;
window.__lastVolume = 0;

var VT = window.VT = {
    capture: null,      // {stream, ctx, source, node, sink, sourceType, closed, onLost}
    ws: null,
    wsTimer: null,
    wsBackoff: 500,
    wantStream: false,
    meterTimer: null,
    lastRunning: null,
    resumeTried: false,
};

function vtSetBox(elemId, text) {
    var el = document.querySelector('#' + elemId + ' textarea, #' + elemId + ' input');
    if (el && el.value !== text) el.value = text;
}
function setStatus(text) { vtSetBox('browser-status', text); }
function setMainStatus(text) { vtSetBox('status-text', text); }
function vtSessionData() { return document.getElementById('session-data'); }
function vtSession() { var d = vtSessionData(); return d ? d.dataset.session : null; }

// Peak-hold state: the highest dB seen in the last PEAK_HOLD_MS milliseconds.
var peakDb = -60;
var peakHoldTimer = null;
var PEAK_HOLD_MS = 2000;   // hold peak for 2 s before decaying

function updateVolumeMeter(rmsLevel) {
    var bar          = document.getElementById('volume-meter-bar');
    var tline        = document.getElementById('volume-meter-threshold-line');
    var peakLine     = document.getElementById('volume-meter-peak-line');
    var label        = document.getElementById('volume-meter-db-label');
    var thresholdLabel = document.getElementById('volume-meter-threshold-label');
    if (!bar) return;

    // Convert RMS → dB (dBFS), clamp to display range
    var db = rmsLevel > 0 ? 20 * Math.log10(rmsLevel) : -60;
    db = Math.max(-60, Math.min(0, db));
    var percent = (db + 60) / 60 * 100;

    // ── Bar: NO smoothing — shows exactly what the VAD sees ──────────────────
    bar.style.width = percent + '%';

    // Get threshold dB from slider (direct dB value)
    var tDb = -30;
    var slider = document.querySelector('#vad-threshold-slider input[type="range"]');
    if (slider) tDb = parseFloat(slider.value) || -30;
    tDb = Math.max(-60, Math.min(0, tDb));

    // Colour: red if above threshold (VAD is active), green if below
    var aboveThreshold = db >= tDb;
    bar.style.background = aboveThreshold ? '#ff5555' : '#44cc44';

    var led = document.getElementById('vad-led');
    if (led) {
        led.style.background = aboveThreshold ? '#44ff44' : '#888';
        led.style.boxShadow = aboveThreshold ? '0 0 5px #44ff44' : 'none';
    }

    // ── dB label ──────────────────────────────────────────────────────────────
    if (label) label.textContent = db.toFixed(1) + ' dB';

    // ── Threshold line ────────────────────────────────────────────────────────
    if (tline) {
        var tPercent = (tDb + 60) / 60 * 100;
        tline.style.left = tPercent + '%';
        if (thresholdLabel) {
            thresholdLabel.textContent = 'threshold: ' + tDb.toFixed(0) + ' dB';
        }
    }

    // ── Peak-hold line ────────────────────────────────────────────────────────
    // Stays at the maximum dB seen; resets after PEAK_HOLD_MS of no new peaks.
    if (db > peakDb) {
        peakDb = db;
        if (peakHoldTimer) clearTimeout(peakHoldTimer);
        peakHoldTimer = setTimeout(function() {
            peakDb = -60;
            if (peakLine) peakLine.style.left = '0%';
        }, PEAK_HOLD_MS);
    }
    if (peakLine) {
        var peakPercent = (peakDb + 60) / 60 * 100;
        peakLine.style.left = Math.max(0, peakPercent - 0.5) + '%'; // centre the 2px line
    }
}

// Gradio may inject this script after DOMContentLoaded has already fired
// (and renders its components later still), so boot on a short retry loop
// instead of relying on that event.
function vtWhenReady(fn) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fn);
    } else {
        fn();
    }
}

vtWhenReady(function() {
    var tries = 0;
    var t = setInterval(function() {
        var slider = document.querySelector('#vad-threshold-slider input[type="range"]');
        if (slider) {
            clearInterval(t);
            slider.addEventListener('input', function() {
                updateVolumeMeter(window.__lastVolume || 0);
            });
        } else if (++tries > 300) {
            clearInterval(t);
        }
    }, 100);
});

// ── Hardware mic level polling ─────────────────────────────────────────────
window.startHwLevelPolling = function() {
    if (window.__hwLevelInterval) return;
    var session = vtSession();
    if (!session) return;
    window.__hwLevelInterval = setInterval(function() {
        fetch('/mic_level/' + session)
            .then(function(r) { return r.json(); })
            .then(function(d) { if (typeof d.rms === 'number') updateVolumeMeter(d.rms); })
            .catch(function() {});
    }, 50);
};

window.stopHwLevelPolling = function() {
    if (window.__hwLevelInterval) {
        clearInterval(window.__hwLevelInterval);
        window.__hwLevelInterval = null;
    }
    updateVolumeMeter(0);
};

// ── Audio source detection ─────────────────────────────────────────────────
// Mirrors whichever "Audio Source" radio option is currently checked.
window.__getAudioSourceType = function() {
    var dispRadio = document.querySelector('input[value="browser_display"]');
    if (dispRadio && dispRadio.checked) return 'display';
    return 'mic';
};
function vtBrowserModeSelected() {
    var micRadio = document.querySelector('input[value="browser_mic"]');
    var dispRadio = document.querySelector('input[value="browser_display"]');
    return (micRadio && micRadio.checked) || (dispRadio && dispRadio.checked);
}

// ── 16 kHz resampler (shared by the AudioWorklet and the fallback path) ─────
// Streaming linear interpolation with the fractional position carried across
// blocks (the old per-block resampler restarted its phase on every block,
// adding a small glitch each time), after a short moving-average low-pass to
// limit aliasing. Emits 40 ms Int16 chunks plus the peak 10 ms-frame RMS
// (exactly what the server VAD measures) for the level meter.
var VT_RESAMPLER_SRC = `
class VTResampler {
    constructor(inRate) {
        this.ratio = inRate / 16000;
        this.t = 0;
        this.prev = 0;
        this.taps = Math.max(1, Math.min(8, Math.round(this.ratio)));
        this.hist = new Float32Array(this.taps);
        this.hi = 0;
        this.sum = 0;
        this.scratch = null;
        this.out = new Int16Array(640);
        this.n = 0;
        this.frameAcc = 0;
        this.frameN = 0;
        this.peak = 0;
    }
    push(input, emit) {
        var L = input.length;
        if (L < 2) return;
        var x = input;
        if (this.taps > 1) {
            if (!this.scratch || this.scratch.length !== L) this.scratch = new Float32Array(L);
            x = this.scratch;
            for (var i = 0; i < L; i++) {
                this.sum += input[i] - this.hist[this.hi];
                this.hist[this.hi] = input[i];
                this.hi = (this.hi + 1) % this.taps;
                if (this.hi === 0) {  // re-sum now and then so float error can't drift
                    var s = 0;
                    for (var k = 0; k < this.taps; k++) s += this.hist[k];
                    this.sum = s;
                }
                x[i] = this.sum / this.taps;
            }
        }
        var t = this.t;
        while (t < L - 1) {
            var i1 = Math.floor(t);
            var f = t - i1;
            var a = i1 < 0 ? this.prev : x[i1];
            var v = a + (x[i1 + 1] - a) * f;
            if (v > 1) v = 1; else if (v < -1) v = -1;
            this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
            this.frameAcc += v * v;
            if (++this.frameN === 160) {
                var r = Math.sqrt(this.frameAcc / 160);
                if (r > this.peak) this.peak = r;
                this.frameAcc = 0;
                this.frameN = 0;
            }
            if (this.n === this.out.length) {
                emit(this.out.slice(0), this.peak);
                this.n = 0;
                this.peak = 0;
            }
            t += this.ratio;
        }
        this.t = t - L;
        this.prev = x[L - 1];
    }
}
`;

var VT_WORKLET_SRC = VT_RESAMPLER_SRC + `
class VTCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        var port = this.port;
        this.rs = new VTResampler(sampleRate);
        this.emit = function(pcm, peak) { port.postMessage({ pcm: pcm.buffer, peak: peak }, [pcm.buffer]); };
    }
    process(inputs) {
        var ch = inputs[0] && inputs[0][0];
        if (ch) this.rs.push(ch, this.emit);
        return true;
    }
}
registerProcessor('vt-capture', VTCaptureProcessor);
`;

var VTResamplerMain = (new Function(VT_RESAMPLER_SRC + '; return VTResampler;'))();

// ── Capture (shared by streaming and the mic test) ─────────────────────────
// sourceType: 'mic' (getUserMedia, this device's microphone) or 'display'
// (getDisplayMedia — a shared browser tab/window/screen, with its audio —
// e.g. a Discord web tab, or on Windows/ChromeOS, whole-system audio).
function vtGetMedia(sourceType) {
    if (sourceType === 'display') {
        // video:true is required by the getDisplayMedia spec even though we
        // discard the track immediately below — only the audio is used.
        return navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
            .then(function(stream) {
                stream.getVideoTracks().forEach(function(t) { stream.removeTrack(t); t.stop(); });
                if (stream.getAudioTracks().length === 0) {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                    throw new Error("No audio shared — check 'Share audio' in the picker");
                }
                return stream;
            });
    }
    return navigator.mediaDevices.getUserMedia({ audio: true });
}

async function vtOpenCapture(sourceType, onChunk) {
    var stream = await vtGetMedia(sourceType);
    var AC = window.AudioContext || window.webkitAudioContext;
    var ctx = new AC({ latencyHint: 'interactive' });
    var source = ctx.createMediaStreamSource(stream);
    // Muted sink: keeps the graph pulling audio without playing it back.
    var sink = ctx.createGain();
    sink.gain.value = 0;
    sink.connect(ctx.destination);
    var node = null;
    if (ctx.audioWorklet && window.AudioWorkletNode) {
        try {
            var url = URL.createObjectURL(new Blob([VT_WORKLET_SRC], { type: 'application/javascript' }));
            await ctx.audioWorklet.addModule(url);
            URL.revokeObjectURL(url);
            node = new AudioWorkletNode(ctx, 'vt-capture', {
                numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]
            });
            node.port.onmessage = function(e) { onChunk(e.data.pcm, e.data.peak); };
        } catch (err) {
            console.warn('[VT] AudioWorklet unavailable, using ScriptProcessor', err);
            node = null;
        }
    }
    if (!node) {
        var rs = new VTResamplerMain(ctx.sampleRate);
        node = ctx.createScriptProcessor(2048, 1, 1);
        node.onaudioprocess = function(e) {
            rs.push(e.inputBuffer.getChannelData(0), function(pcm, peak) { onChunk(pcm.buffer, peak); });
        };
    }
    source.connect(node);
    node.connect(sink);
    return { stream: stream, ctx: ctx, source: source, node: node, sink: sink,
             sourceType: sourceType, closed: false, onLost: null };
}

function vtCloseCapture(cap) {
    if (!cap || cap.closed) return;
    cap.closed = true;
    try { if (cap.node.port) cap.node.port.onmessage = null; } catch (e) {}
    try { cap.node.onaudioprocess = null; } catch (e) {}
    try { cap.source.disconnect(); } catch (e) {}
    try { cap.node.disconnect(); } catch (e) {}
    cap.stream.getTracks().forEach(function(t) { t.onended = null; try { t.stop(); } catch (e) {} });
    try { cap.ctx.onstatechange = null; cap.ctx.close(); } catch (e) {}
}

// Detect the source going away, and keep the AudioContext running.
function vtWatchCapture(cap) {
    cap.stream.getAudioTracks().forEach(function(t) {
        t.onended = function() {
            if (cap.onLost) cap.onLost(cap.sourceType === 'display' ? 'sharing stopped' : 'microphone disconnected');
        };
    });
    cap.ctx.onstatechange = function() { vtEnsureRunning(cap); };
    vtEnsureRunning(cap);
}

function vtEnsureRunning(cap) {
    if (!cap || cap.closed) return;
    var st = cap.ctx.state;
    if (st === 'running') {
        if (VT.pausedMsg) { VT.pausedMsg = false; setStatus(VT.wantStream ? 'Streaming...' : 'Mic test active...'); }
        return;
    }
    if (st === 'closed') return;
    // 'suspended' / 'interrupted': the browser paused audio (autoplay policy
    // after a reload, OS audio interruption, …). Try to resume; if that needs
    // a user gesture, ask for one.
    cap.ctx.resume().catch(function() {});
    setTimeout(function() {
        if (cap.closed || cap.ctx.state === 'running') return;
        VT.pausedMsg = true;
        setStatus('⏸ The browser paused audio capture — click anywhere on this page to resume');
        var resume = function() {
            document.removeEventListener('pointerdown', resume, true);
            document.removeEventListener('keydown', resume, true);
            if (!cap.closed) cap.ctx.resume().catch(function() {});
        };
        document.addEventListener('pointerdown', resume, true);
        document.addEventListener('keydown', resume, true);
    }, 400);
}

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', function() {
        var cap = VT.capture;
        if (!cap || cap.closed) return;
        var tracks = cap.stream.getAudioTracks();
        var dead = tracks.length === 0 || tracks.every(function(t) { return t.readyState === 'ended'; });
        if (dead && cap.onLost) cap.onLost('microphone disconnected');
    });
}
document.addEventListener('visibilitychange', function() {
    if (!document.hidden && VT.capture) vtEnsureRunning(VT.capture);
});

// ── Level meter ────────────────────────────────────────────────────────────
function vtNoteLevel(peak) {
    if (!(peak <= window.__lastVolume)) window.__lastVolume = peak;
}
function vtStartMeter() {
    if (VT.meterTimer) return;
    VT.meterTimer = setInterval(function() {
        updateVolumeMeter(window.__lastVolume || 0);
        window.__lastVolume = 0;
    }, 80);
}
function vtStopMeter() {
    if (VT.meterTimer) { clearInterval(VT.meterTimer); VT.meterTimer = null; }
    window.__lastVolume = 0;
    updateVolumeMeter(0);
}

// ── Audio WebSocket (auto-reconnecting) ────────────────────────────────────
function vtConnectWs() {
    if (!VT.wantStream || VT.ws) return;
    var d = vtSessionData();
    if (!d || !d.dataset.wsPath) { VT.wsTimer = setTimeout(vtConnectWs, 250); return; }
    var proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var ws = new WebSocket(proto + '//' + window.location.host + d.dataset.wsPath);
    ws.binaryType = 'arraybuffer';
    VT.ws = ws;
    ws.onopen = function() {
        VT.wsBackoff = 500;
        if (VT.capture) setStatus('Streaming...');
    };
    ws.onmessage = function(e) {
        if (typeof e.data !== 'string') return;
        var msg;
        try { msg = JSON.parse(e.data); } catch (x) { return; }
        if (msg.type === 'stopped') {
            vtStopStreaming(msg.reason || 'Stopped');
            if (msg.reason) setMainStatus(msg.reason);
        } else if (msg.type === 'replaced') {
            vtStopStreaming('Audio moved to another open tab of this session');
        }
    };
    ws.onclose = function() {
        if (VT.ws === ws) VT.ws = null;
        if (!VT.wantStream) return;
        // Network blip / proxy recycle / server restart: keep capturing and
        // reconnect. Audio produced while disconnected is dropped, not
        // buffered — a live tool shouldn't replay the past.
        setStatus('Connection lost — reconnecting…');
        clearTimeout(VT.wsTimer);
        VT.wsTimer = setTimeout(vtConnectWs, VT.wsBackoff);
        VT.wsBackoff = Math.min(VT.wsBackoff * 2, 5000);
    };
}

function vtCloseWs() {
    clearTimeout(VT.wsTimer);
    var ws = VT.ws;
    VT.ws = null;
    if (ws) {
        ws.onclose = null;
        ws.onmessage = null;
        try { ws.close(); } catch (e) {}
    }
}

// ── Streaming ──────────────────────────────────────────────────────────────
function vtStartStreaming(sourceType, auto) {
    if (VT.wantStream) {
        // Already streaming from this tab: just make sure the socket is up.
        vtConnectWs();
        return;
    }
    if (window.__micTestActive) window.stopBrowserMicTest();
    VT.wantStream = true;
    window.__audioStreamActive = true;
    setStatus(sourceType === 'display'
        ? 'Choose a tab/window/screen to share (tick "Share audio")…'
        : 'Starting microphone…');
    vtConnectWs();
    vtOpenCapture(sourceType, function(buf, peak) {
        vtNoteLevel(peak);
        var ws = VT.ws;
        // Only send when the socket keeps up: if ~1 s is already waiting in
        // the send buffer, drop this chunk instead of letting audio pile up.
        if (ws && ws.readyState === 1 && ws.bufferedAmount < 32000) ws.send(buf);
    }).then(function(cap) {
        if (!VT.wantStream) { vtCloseCapture(cap); return; }
        VT.capture = cap;
        cap.onLost = vtSourceLost;
        vtWatchCapture(cap);
        vtStartMeter();
        if (VT.ws && VT.ws.readyState === 1) setStatus('Streaming...');
    }).catch(function(err) {
        VT.wantStream = false;
        window.__audioStreamActive = false;
        vtCloseWs();
        var msg = (sourceType === 'display' ? 'Capture error: ' : 'Mic error: ') + ((err && err.message) || err);
        if (auto) msg += ' — press ▶️ Start to reconnect the microphone';
        setStatus(msg);
        setMainStatus(msg);
        // Nothing will stream, so don't leave the session running without a
        // source. (After an automatic re-attach attempt the server's own
        // watchdog handles it, giving you time to press Start.)
        if (!auto) vtReportSourceEnded('could not open the audio source');
    });
}

function vtStopStreaming(text) {
    VT.wantStream = false;
    window.__audioStreamActive = false;
    vtCloseWs();
    vtCloseCapture(VT.capture);
    VT.capture = null;
    vtStopMeter();
    setStatus(text || 'Stopped');
}

function vtReportSourceEnded(why) {
    var ws = VT.ws;
    if (ws && ws.readyState === 1) {
        try { ws.send(JSON.stringify({ type: 'source_ended', reason: why })); return; } catch (e) {}
    }
    var slug = vtSession();
    if (slug) {
        fetch('/session_stop/' + encodeURIComponent(slug) + '?reason=' + encodeURIComponent(why),
              { method: 'POST', keepalive: true }).catch(function() {});
    }
}

// The capture's track ended: mic unplugged, permission revoked, or
// "Stop sharing" clicked. Stop the session right away.
function vtSourceLost(why) {
    if (!VT.wantStream) {
        if (window.__micTestActive) {
            window.stopBrowserMicTest();
            setStatus('🔌 ' + why);
        }
        return;
    }
    vtReportSourceEnded(why);
    VT.wantStream = false;
    window.__audioStreamActive = false;
    vtCloseCapture(VT.capture);
    VT.capture = null;
    vtStopMeter();
    setTimeout(vtCloseWs, 500);  // let the source_ended message go out first
    var msg = '🔌 Audio source lost (' + why + ') — session stopped';
    setStatus(msg);
    setMainStatus(msg);
}

// Called (as Gradio js=) when ▶️ Start is clicked, before the server starts.
window.startBrowserStreaming = function() {
    if (!vtBrowserModeSelected()) return;
    vtStartStreaming(window.__getAudioSourceType(), false);
};

// Called after the server's start handler returns: if it failed, stop the
// audio we started for it.
window.vtAfterStart = function(status) {
    if (typeof status === 'string' && status.indexOf('❌') === 0 && VT.wantStream) {
        vtStopStreaming('Stopped');
    }
};

window.stopBrowserStreaming = function() {
    if (!VT.wantStream && !VT.capture) return;
    vtStopStreaming('Stopped');
};

// ── Browser mic test (meter only, no WebSocket) ───────────────────────────
window.startBrowserMicTest = function() {
    if (VT.wantStream || window.__micTestActive) return;
    window.__micTestActive = true;
    vtOpenCapture(window.__getAudioSourceType(), function(buf, peak) { vtNoteLevel(peak); })
        .then(function(cap) {
            if (!window.__micTestActive) { vtCloseCapture(cap); return; }
            VT.capture = cap;
            cap.onLost = vtSourceLost;
            vtWatchCapture(cap);
            vtStartMeter();
            setStatus('Mic test active...');
        })
        .catch(function(err) {
            window.__micTestActive = false;
            setStatus('Mic error: ' + ((err && err.message) || err));
        });
};

window.stopBrowserMicTest = function() {
    if (!window.__micTestActive) return;
    window.__micTestActive = false;
    if (!VT.wantStream) {
        vtCloseCapture(VT.capture);
        VT.capture = null;
        vtStopMeter();
    }
};

// ── Display + session-state polling ────────────────────────────────────────
// Plain fetch() against /display_data (never through Gradio's queue). Also
// carries the session's running state, so every open tab notices a stop —
// including an automatic one when the audio source is lost — and a reloaded
// page can re-attach its microphone.
function vtRender(d) {
    var box = document.getElementById('vt-display');
    if (box && typeof d.html === 'string') {
        var empty = !d.recognized && !d.translated;
        var lines = box.querySelector('.vt-lines');
        if (empty && lines && box.__vtHtml !== null) {
            // Fade out what's on screen as one block (text + outline);
            // don't swap in the empty version, which would drop the text
            // instantly and leave only the fade of an empty block.
            lines.style.opacity = '0';
            box.__vtHtml = null;
        } else if (!empty && box.__vtHtml !== d.html) {
            box.innerHTML = d.html;
            box.__vtHtml = d.html;
        } else if (empty && !lines) {
            box.innerHTML = d.html;
            var l = box.querySelector('.vt-lines');
            if (l) l.style.opacity = '0';
            box.__vtHtml = null;
        }
    }
    vtSetBox('recognized-output-text', d.recognized || '');
    vtSetBox('translated-output-text', d.translated || '');
}

function vtHandleState(d) {
    if (typeof d.running !== 'boolean') return;
    var browserMode = d.audio_mode === 'browser_mic' || d.audio_mode === 'browser_display';
    if (VT.lastRunning === true && d.running === false) {
        // Stopped — by Stop in this or another tab, or automatically.
        if (VT.wantStream) vtStopStreaming(d.stop_reason || 'Stopped');
        if (d.stop_reason) setMainStatus(d.stop_reason);
    }
    if (VT.lastRunning === null && d.running && browserMode && !VT.wantStream && !d.audio_client) {
        vtAutoResume(d.audio_mode);
    }
    VT.lastRunning = d.running;
}

function vtAutoResume(mode) {
    if (VT.resumeTried) return;
    VT.resumeTried = true;
    if (mode === 'browser_display') {
        var m = '⚠️ This session is running, but the page was reloaded so tab/screen ' +
                'sharing ended. Press ▶️ Start to share again.';
        setStatus(m);
        setMainStatus(m);
        return;
    }
    setStatus('Page reloaded — reconnecting the microphone…');
    vtStartStreaming('mic', true);
}

function vtPoll() {
    var slug = vtSession();
    if (!slug) { VT.pollTimer = setTimeout(vtPoll, 250); return; }
    fetch('/display_data/' + encodeURIComponent(slug), { cache: 'no-store' })
        .then(function(r) { return r.json(); })
        .then(function(d) { vtRender(d); vtHandleState(d); })
        .catch(function() {})
        .finally(function() {
            // Slow down while the tab is hidden; the session itself is unaffected.
            VT.pollTimer = setTimeout(vtPoll, document.hidden ? 1000 : 100);
        });
}

// ── Logs polling ───────────────────────────────────────────────────────────
window.__logsInterval = null;

window.startAllPolling = function() {
    var session = vtSession();
    if (!session) return false;
    if (!VT.pollTimer) vtPoll();
    if (!window.__logsInterval) {
        window.__logsInterval = setInterval(function() {
            fetch('/logs_data/' + session)
                .then(function(r) { return r.json(); })
                .then(function(d) { if (d.logs !== undefined) vtSetBox('log_output', d.logs); })
                .catch(function() {});
        }, 2000);
    }
    return true;
};

window.stopAllPolling = function() {
    if (VT.pollTimer) { clearTimeout(VT.pollTimer); VT.pollTimer = null; }
    if (window.__logsInterval) { clearInterval(window.__logsInterval); window.__logsInterval = null; }
};

// Start polling as soon as Gradio has injected the session-data div.
vtWhenReady(function() {
    var boot = setInterval(function() {
        if (window.startAllPolling()) clearInterval(boot);
    }, 100);
});
</script>
"""


# dots_or_stars, is_whisper_hallucination, ArgosTranslator, WhisperRecognizer
# now live in recognizers.py — imported above.

# FastVAD now lives in vad.py; MOONSHINE_LANGUAGES/MoonshineRecognizer now
# live in recognizers.py — both imported above.

# SubtitleManager now lives in subtitles.py — imported above.

# ── VoiceTranslatorApp ────────────────────────────────────────────────────────
class VoiceTranslatorApp:
    # Default settings – overridden by saved settings on load
    DEFAULT_SETTINGS = {
        "model_path": "",
        "microphone": 0,
        "vosk_model": "",
        "audio_mode": "hardware",
        "enable_translation": False,
        "display_interim": True,
        "translation_mode": "argos",
        "source_language": "en",
        "target_language": "es",
        "font_family": "Arial",
        "recognized_font_size": 48,
        "translated_font_size": 32,
        "recognized_color": "#FFFFFF",
        "translated_color": "#CCCCCC",
        "background_color": "#000000",
        "text_alignment": "center",
        "vertical_alignment": "middle",  # top | middle | bottom (bottom = classic subtitles)
        "translation_position": "after",
        "recognition_engine": "vosk",
        "whisper_host": "http://localhost:9000",
        "whisper_api_key": "",
        "whisper_model": "whisper-large-v3",
        "whisper_language": "en",
        "whisper_translate_host": "http://localhost:9000",
        "whisper_translate_api_key": "",
        "whisper_translate_model": "whisper-large-v3",
        "argos_source_lang": "en",
        "argos_target_lang": "es",
        "libretranslate_host": "",
        "libretranslate_api_key": "",
        "fade_timeout": 5.0,
        "ai_host": "http://localhost:11434",
        "ai_api_key": "",
        "ai_model": "",
        # Advanced/developer overrides for AI-translation prompt + endpoint —
        # blank means "use the built-in default", see AI_TRANSLATION.md.
        "ai_translation_prompt_template": "",
        "ai_endpoint_url": "",
        "ai_request_body_template": "",
        "ai_response_text_path": "",
        "outline_width": 0,
        "outline_color": "#000000",
        "translated_outline_width": 0,
        "translated_outline_color": "#000000",
        "custom_font": "",
        "whisper_temperature": 0.0,
        "whisper_best_of": 5,
        "whisper_beam_size": 5,
        "whisper_patience": 1.0,
        "whisper_length_penalty": 1.0,
        "whisper_suppress_tokens": "-1",
        "whisper_initial_prompt": "",
        "whisper_condition_on_previous_text": False,  # was True — this feeds each new segment's decoding the *previous* segment's text, which biases it toward continuing whatever pattern it just produced. Once it hallucinates "Thank you." once, this makes it more likely to keep producing outro-style filler on the next ambiguous/quiet segment too. Off by default now; turn on if you specifically want better cross-segment context for long continuous speech and hallucinations aren't a problem for your setup.
        "whisper_temperature_increment_on_fallback": 0.2,
        "whisper_no_speech_threshold": 0.6,
        "whisper_logprob_threshold": -1.0,
        "whisper_compression_ratio_threshold": 2.4,
        # Advanced/developer overrides for a non-OpenAI-compatible Whisper
        # server — blank means "use the auto-derived /audio/transcriptions
        # path and standard verbose_json response parsing", same pattern as
        # the AI-translation overrides. See AI_TRANSLATION.md.
        "whisper_endpoint_url": "",
        "whisper_response_text_path": "",
        "whisper_translate_temperature": 0.0,
        "whisper_translate_best_of": 5,
        "whisper_translate_beam_size": 5,
        "whisper_translate_patience": 1.0,
        "whisper_translate_length_penalty": 1.0,
        "whisper_translate_suppress_tokens": "-1",
        "whisper_translate_initial_prompt": "",
        "whisper_translate_condition_on_previous_text": False,  # see whisper_condition_on_previous_text above — same hallucination-loop concern applies here
        "whisper_translate_temperature_increment_on_fallback": 0.2,
        "whisper_translate_no_speech_threshold": 0.6,
        "whisper_translate_logprob_threshold": -1.0,
        "whisper_translate_compression_ratio_threshold": 2.4,
        "whisper_translate_endpoint_url": "",
        "whisper_translate_response_text_path": "",
        # VAD — threshold + end-of-speech delay are both hot-reloadable
        "vad_threshold": -30.0,  # dB — the slider value is now in dB directly
        # Sounds with less actual speech than this (desk knocks, clicks,
        # coughs) are never sent to Whisper — see FastVAD._set_min_speech.
        "vad_min_speech_ms": 200,
        "vad_end_silence_ms": 300,  # ms of silence before dispatching (Whisper/Moonshine) — was 80ms, too short: natural mid-sentence pauses (breathing, thinking) routinely exceed that, so Whisper got flooded with tiny fragmented clips, which is exactly when it hallucinates fillers like "thank you"
        # Moonshine (moonshine-voice package)
        "moonshine_language": "en",
        "moonshine_cache_dir": "moonshine_models",
        # Subtitle display
        "subtitle_mode": "instant",  # "instant" | "buffered"
        "subtitle_cps": 21,  # characters per second
        "subtitle_max_lines": 2,  # sentences per chunk in buffered mode
        "noise_filter_threshold": 0.0,
        # Auto-stop a running session when its audio source goes away: a
        # server device that stops delivering audio (unplugged), or a
        # browser mic/tab stream that stops arriving. 0 disables. See
        # SESSIONS.md → "Audio watchdog".
        "audio_watchdog_seconds": 8,
        # Whisper live mode (see RECOGNITION_QUALITY.md → "Live latency")
        "whisper_low_latency": True,  # greedy decoding (beam 1 / best-of 1)
        "whisper_interim": True,  # live partial captions while still speaking
        "whisper_max_segment_s": 6.0,  # cut continuous speech into pieces this long
    }

    def __init__(self, slug: str):
        self.session_active = True
        self.display_running = True
        self.slug = slug
        self.last_active = time.time()

        self.audio_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        # Translation runs on its own thread so a slow translator never holds
        # back the recognized text (see _update_display_loop).
        self._translate_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.is_monitoring = False  # mic-test mode (level only, no recognition)

        self.recognizer = None
        self.model = None
        self._vosk_model_path: str | None = None  # which path self.model is a shared reference to — needed by _unload_vosk to release the right cache entry
        self.whisper_recognizer: WhisperRecognizer | None = None
        self.moonshine_recognizer: MoonshineRecognizer | None = None
        self.argos_translator: ArgosTranslator | None = None
        self.translation_service: TranslationService | None = None
        self.stream = None
        self._monitor_stream = None
        self.vad: FastVAD | None = None
        self.monitor_level: float = 0.0  # latest RMS from hardware mic
        self.whisper_worker: LiveWhisperWorker | None = None
        self._whisper_translator: WhisperRecognizer | None = None
        self._last_interim_at = 0.0
        self._last_backlog_warning = 0.0

        # Audio-source liveness (see check_audio_watchdog)
        self._started_at = 0.0
        self._last_audio_at = 0.0
        # Why the session last stopped + a counter the page polls, so an
        # automatic stop (source disconnected) is shown in every open tab.
        self.stop_reason: str | None = None
        self.state_seq = 0

        # The one browser tab currently streaming audio into this session:
        # (connection id, WebSocket, event loop). A newer tab replaces it.
        self._audio_client: tuple | None = None
        self._audio_client_lock = threading.Lock()

        # Serializes start/stop/monitor transitions. Without this, Gradio's
        # concurrency settings (interface.queue(default_concurrency_limit=None))
        # allow two clicks on this session — e.g. a fast Start-then-Stop, or a
        # double-click — to run start_recognition()/stop_recognition() on two
        # different threads AT THE SAME TIME. That's a real crash vector: Vosk's
        # recognizer/model are C++ objects, and one thread deleting them (stop)
        # while another thread is mid-call into them (start's processing loop,
        # or vice versa) is undefined behavior — it can hard-crash the whole
        # Python process, not just raise a catchable exception. That matches
        # "the service restarts when I start and stop a model."
        self._control_lock = threading.RLock()

        self.vosk_audio_buffer = bytearray()
        self.last_audio_chunk: bytes | None = None

        self.logger = Logger(session_id=slug)

        # Build settings: start from defaults, overlay saved settings
        self.settings: dict = dict(self.DEFAULT_SETTINGS)
        self.settings.update(load_saved_settings(slug))
        # Migrate old "browser" audio_mode value → "browser_mic" (Phase 3
        # split the old single browser mode into mic vs tab/system audio).
        if self.settings.get("audio_mode") == "browser":
            self.settings["audio_mode"] = "browser_mic"
        # The popout id lives in settings so it's saved per session and an
        # OBS browser source keeps working across restarts. Generate (and
        # save) one the first time a session exists.
        if not self.settings.get("popout_id"):
            self.settings["popout_id"] = secrets.token_urlsafe(16)
            persist_settings(slug, self.settings)

        # Subtitle manager — settings applied dynamically
        self.subtitles = SubtitleManager(
            mode=self.settings["subtitle_mode"],
            cps=self.settings["subtitle_cps"],
            max_lines=self.settings["subtitle_max_lines"],
            fade_timeout=self.settings["fade_timeout"],
        )

        if ARGOS_AVAILABLE:
            self.argos_translator = ArgosTranslator(logger=self.logger)

        self.display_thread = threading.Thread(
            target=self._update_display_loop, daemon=True
        )
        self.display_thread.start()
        self.translate_thread = threading.Thread(
            target=self._translate_loop, daemon=True
        )
        self.translate_thread.start()

    @property
    def popout_id(self) -> str:
        return self.settings.get("popout_id", "")

    @popout_id.setter
    def popout_id(self, value: str):
        self.settings["popout_id"] = value
        persist_settings(self.slug, self.settings)

    # ── Validation helpers ────────────────────────────────────────────────────
    def is_repetitive_garbage(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) > 10:
            counts: dict = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            if max(counts.values()) > 5:
                self.logger.log(f"Filtered repetitive words: {text}", level="debug")
                return True
        cleaned = re.sub(r"[\s.,!?]", "", text)
        if len(cleaned) > 10:
            max_run = current_run = 1
            for i in range(1, len(cleaned)):
                if cleaned[i] == cleaned[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            if max_run > 5:
                self.logger.log(f"Filtered character run: {text}", level="debug")
                return True
        return False

    def is_valid_transcription(self, text: str) -> bool:
        if re.search(r"<\|.*?\|>", text):
            self.logger.log(f"Filtered (timestamp token): {text}", level="debug")
            return False
        if len(text) > 500:
            self.logger.log(f"Filtered (too long): {text}", level="debug")
            return False
        return not self.is_repetitive_garbage(text)

    def _moonshine_result(self, text: str, is_final: bool):
        """Called by MoonshineRecognizer's listener thread when speech is detected."""
        if not text or not self.is_running:
            return
        if is_final:
            if self.is_valid_transcription(text):
                self.result_queue.put(("final", text))
        elif self.settings.get("display_interim"):
            self.result_queue.put(("interim", text))

    # ── Dynamic settings helpers ──────────────────────────────────────────────
    def apply_vad_settings(self):
        """Push latest threshold + end_silence + noise filter into the running VAD."""
        if self.vad:
            self.vad.update_threshold(self.settings.get("vad_threshold", -30.0))
            self.vad.update_end_silence_ms(self.settings.get("vad_end_silence_ms", 300))
            self.vad.update_max_segment_ms(self._max_segment_ms())
            self.vad.update_min_speech_ms(self.settings.get("vad_min_speech_ms", 200))
            thresh = self.settings.get("noise_filter_threshold", 0.0)
            self.vad.update_noise_filter_threshold(thresh)

    def apply_subtitle_settings(self):
        """Push latest subtitle settings into SubtitleManager."""
        self.subtitles.update_settings(
            mode=self.settings.get("subtitle_mode", "instant"),
            cps=self.settings.get("subtitle_cps", 21),
            max_lines=self.settings.get("subtitle_max_lines", 2),
            fade_timeout=self.settings.get("fade_timeout", 5.0),
        )

    # ── Mic monitoring (test mic, level display without recognition) ──────────
    def _monitor_callback(self, indata, frames, time_info, status):
        samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        frame_len = 160
        n_frames = len(samples) // frame_len
        if n_frames > 0:
            peak_rms = 0.0
            for i in range(n_frames):
                frame = samples[i * frame_len : (i + 1) * frame_len]
                rms = float(np.sqrt(np.mean(frame**2)))
                if rms > peak_rms:
                    peak_rms = rms
            self.monitor_level = peak_rms
        else:
            self.monitor_level = float(np.sqrt(np.mean(samples**2)))

    def start_mic_monitor(self) -> str:
        if not self._control_lock.acquire(timeout=2.0):
            return "⏳ Still starting/stopping — try again in a moment"
        try:
            if self.is_running:
                return "⚠️ Recognition is active — level already visible in the meter"
            if self.is_monitoring:
                return "Already monitoring"
            mic = self.settings.get("microphone")
            if mic is None:
                return "❌ No microphone selected"
            try:
                # 16 kHz if the device supports it, else its own rate
                # resampled to 16 kHz (see audio_input.py). 30 ms blocks.
                self._monitor_stream = open_input_stream(
                    mic, self._monitor_callback, logger=self.logger
                )
                self._monitor_stream.start()
                self.is_monitoring = True
                return "🎙️ Mic monitor active — adjust threshold until line sits above background noise"
            except Exception as exc:
                return f"❌ Monitor error: {exc}"
        finally:
            self._control_lock.release()

    def stop_mic_monitor(self) -> str:
        if not self._control_lock.acquire(timeout=5.0):
            return "⏳ Still busy — try Stop again in a moment"
        try:
            if self._monitor_stream:
                try:
                    self._monitor_stream.stop()
                    self._monitor_stream.close()
                except Exception:
                    pass
                self._monitor_stream = None
            self.is_monitoring = False
            self.monitor_level = 0.0
            return "⏹️ Monitor stopped"
        finally:
            self._control_lock.release()

    # ── Audio processing ──────────────────────────────────────────────────────
    def audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self._last_audio_at = time.time()
            data = bytes(indata)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            frame_len = 160
            n_frames = len(samples) // frame_len
            if n_frames > 0:
                peak_rms = 0.0
                for i in range(n_frames):
                    frame = samples[i * frame_len : (i + 1) * frame_len]
                    rms = float(np.sqrt(np.mean(frame**2)))
                    if rms > peak_rms:
                        peak_rms = rms
                self.monitor_level = peak_rms
            else:
                self.monitor_level = float(np.sqrt(np.mean(samples**2)))

            self.audio_queue.put(data)  # <-- unbounded queue, never blocks

    # ── Browser audio (WebSocket) ─────────────────────────────────────────────
    def attach_audio_client(self, websocket, loop) -> tuple[int, tuple | None]:
        """
        Make `websocket` this session's browser audio source. Returns
        (connection id, previous client or None). Only one tab streams into a
        session at a time — two tabs feeding the same recognizer would
        interleave their audio — so the newest one wins and the caller tells
        the previous one it was replaced.
        """
        conn_id = id(websocket)
        with self._audio_client_lock:
            previous = self._audio_client
            self._audio_client = (conn_id, websocket, loop)
        return conn_id, previous

    def detach_audio_client(self, conn_id: int):
        with self._audio_client_lock:
            if self._audio_client and self._audio_client[0] == conn_id:
                self._audio_client = None

    @property
    def has_audio_client(self) -> bool:
        return self._audio_client is not None

    def feed_browser_audio(self, data: bytes, conn_id: int):
        # Audio arriving while stopped (or from a tab that has been replaced)
        # is dropped. It used to be queued regardless, so a stopped session
        # could build up minutes of stale audio that the next Start then
        # processed first.
        client = self._audio_client
        if not self.is_running or client is None or client[0] != conn_id:
            return
        if not str(self.settings.get("audio_mode", "")).startswith("browser"):
            return
        self._last_audio_at = time.time()
        self.audio_queue.put(data)
        self.touch()

    def _notify_audio_client(self, payload: dict):
        """Send a JSON control message to the streaming tab, from any thread."""
        client = self._audio_client
        if not client:
            return
        _, ws, loop = client
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(payload)), loop)
        except Exception:
            pass

    def _max_segment_ms(self) -> int:
        if self.settings.get("recognition_engine") != "whisper":
            return 0
        try:
            return int(float(self.settings.get("whisper_max_segment_s", 6.0)) * 1000)
        except (TypeError, ValueError):
            return 6000

    def _get_or_create_vad(self) -> FastVAD:
        if self.vad is None:
            self.vad = FastVAD(
                threshold_db=self.settings.get("vad_threshold", -30.0),
                end_silence_ms=self.settings.get("vad_end_silence_ms", 300),
                noise_filter_threshold=self.settings.get("noise_filter_threshold", 0.0),
                max_segment_ms=self._max_segment_ms(),
                min_speech_ms=self.settings.get("vad_min_speech_ms", 200),
            )
        return self.vad

    def _process_vosk(self, data: bytes):
        """
        Feed audio to Vosk. Noise preprocessing applied first so Vosk receives
        cleaner audio. Vosk's own endpointer handles segmentation.
        """
        if self.recognizer is None:
            # Engine was switched away from vosk (or we're mid-stop) while a
            # block was already in flight to this thread — drop it instead of
            # crashing on a None recognizer.
            return
        vad = self._get_or_create_vad()
        clean = vad.preprocess_block(data)
        self.vosk_audio_buffer.extend(clean)
        if self.recognizer.AcceptWaveform(clean):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                self.logger.log(f"Recognized: {text}", level="info")
                self.result_queue.put(("final", text))
                self.last_audio_chunk = bytes(self.vosk_audio_buffer)
                self.vosk_audio_buffer.clear()
            # Empty final (silence) — let the fade timer clear display naturally
        elif self.settings.get("display_interim"):
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                self.result_queue.put(("interim", partial_text))

    def _process_vad_engine(self, data: bytes):
        """
        VAD + preprocessing pipeline for Whisper and Moonshine.
        process_chunk() handles both noise cleaning and segmentation internally.
        Moonshine receives clean preprocessed audio; Whisper receives clean segments.
        """
        engine = self.settings["recognition_engine"]
        vad = self._get_or_create_vad()

        if engine == "moonshine":
            # Moonshine has its own VAD — feed clean audio continuously
            clean = vad.preprocess_block(data)
            if self.moonshine_recognizer:
                self.moonshine_recognizer.add_audio(clean)
            return

        # Whisper: VAD segments + cleans in one pass. Finished segments go to
        # the ordered live worker (see live_whisper.py).
        worker = self.whisper_worker
        if worker is None:
            return
        rejected_before = vad.rejected
        for speech_bytes in vad.process_chunk(data):
            self.last_audio_chunk = speech_bytes
            worker.submit_final(speech_bytes)
            self._last_interim_at = time.monotonic()
        if vad.rejected != rejected_before:
            self.logger.log(
                "Ignored a short non-speech sound (knock/click) — not sent to Whisper",
                level="debug",
            )

        # Live partial captions for the utterance still in progress — only
        # while the worker is otherwise idle, so they never delay a final.
        if (
            self.settings.get("whisper_interim", True)
            and vad.in_speech
            and vad.current_segment_ms() >= _INTERIM_MIN_AUDIO_MS
            and time.monotonic() - self._last_interim_at >= _INTERIM_INTERVAL_S
            and worker.idle
        ):
            if worker.submit_interim(vad.current_segment()):
                self._last_interim_at = time.monotonic()

    def _apply_noise_filter(self, data: bytes) -> bytes:
        """Legacy shim — delegates to VAD's integrated preprocessor."""
        return self._get_or_create_vad().preprocess_block(data)

    def _clean_whisper_text(self, transcription: str) -> str:
        """Apply the hallucination/garbage filters; '' if the text should be dropped."""
        transcription = (transcription or "").strip()
        if not transcription or dots_or_stars(transcription):
            return ""
        if is_whisper_hallucination(transcription):
            self.logger.log(f"Blocked hallucination: {repr(transcription)}", level="debug")
            return ""
        if not self.is_valid_transcription(transcription):
            self.logger.log("Discarded invalid transcription", level="debug")
            return ""
        return transcription

    def _whisper_transcribe(self, rec: WhisperRecognizer, audio: bytes, interim: bool) -> str:
        """LiveWhisperWorker's transcribe_fn."""
        return rec.transcribe(
            audio,
            language=self.settings.get("whisper_language"),
            # A late interim is useless — give up quickly and let the next one run.
            timeout=10 if interim else 30,
        )

    def _whisper_final(self, text: str, audio: bytes):
        text = self._clean_whisper_text(text)
        if text:
            self.result_queue.put(("final", text, audio))

    def _whisper_interim(self, text: str):
        text = self._clean_whisper_text(text)
        if text and self.is_running:
            self.result_queue.put(("interim", text))

    def _drop_audio_backlog(self):
        """
        Keep recognition live if processing falls behind the incoming audio
        (e.g. a slow CPU running Vosk): rather than working through an
        ever-growing queue of old audio, drop the oldest blocks.
        """
        backlog = self.audio_queue.qsize()
        if backlog <= _AUDIO_BACKLOG_MAX_BLOCKS:
            return
        dropped = 0
        while self.audio_queue.qsize() > _AUDIO_BACKLOG_KEEP_BLOCKS:
            try:
                self.audio_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        now = time.monotonic()
        if now - self._last_backlog_warning > 10:
            self._last_backlog_warning = now
            self.logger.log(
                f"Audio processing fell behind — skipped {dropped} old audio "
                f"blocks to stay live",
                level="warning",
            )

    def process_audio_hardware(self):
        while self.is_running:
            try:
                self._drop_audio_backlog()
                # Timeout matches blocksize (30ms) so we never wait longer than one block
                data = self.audio_queue.get(timeout=0.03)
                engine = self.settings["recognition_engine"]
                if engine == "vosk":
                    self._process_vosk(data)
                else:
                    self._process_vad_engine(data)
            except queue.Empty:
                continue
            except Exception as exc:
                self.logger.log(f"Audio processing error: {exc}", level="error")

    # ── Recognition control ───────────────────────────────────────────────────
    def start_recognition(self, model_path: str | None, microphone_index=None) -> str:
        # Non-blocking: a rapid double-click on Start should be rejected
        # immediately rather than queue up and fire a second start while the
        # first is still loading a model.
        if not self._control_lock.acquire(timeout=0.1):
            return "⏳ Already starting or stopping — please wait a moment"
        try:
            if self.is_running:
                if str(self.settings.get("audio_mode", "")).startswith("browser"):
                    # The page (re)attaches its browser audio on every Start
                    # click, so this is how you resume after a page reload.
                    return "✅ Already running — browser audio re-attached"
                return "⚠️ Already running — press Stop first"

            # Defensive: release any vosk model reference from a previous run
            # that wasn't cleanly stopped. Shouldn't happen given the
            # is_running guard + stop-before-start invariant, but a leaked
            # cache refcount would slowly starve RAM over time, so make sure
            # every start begins from a clean slate regardless.
            if self._vosk_model_path:
                _release_vosk_model(self._vosk_model_path)
                self._vosk_model_path = None
            self.model = None
            self.recognizer = None

            engine = self.settings["recognition_engine"]

            if engine == "vosk":
                if not model_path or not Path(model_path).exists():
                    msg = "❌ Vosk model not found. Place model in vosk_models/"
                    self.logger.log(msg, level="error")
                    return msg
                self.model = _acquire_vosk_model(model_path)
                self._vosk_model_path = model_path
                self.recognizer = KaldiRecognizer(self.model, 16000)
                self.recognizer.SetWords(True)
                self.whisper_recognizer = None
                self.moonshine_recognizer = None

            elif engine == "whisper":
                # Low-latency mode: greedy decoding. Beam search (beam 5 /
                # best-of 5) is several times slower per request on most
                # servers for a small accuracy gain — the wrong trade for
                # live captions.
                low_latency = bool(self.settings.get("whisper_low_latency", True))
                self.whisper_recognizer = WhisperRecognizer(
                    host=self.settings["whisper_host"],
                    api_key=self.settings.get("whisper_api_key") or None,
                    model=self.settings["whisper_model"],
                    logger=self.logger,
                    temperature=self.settings["whisper_temperature"],
                    best_of=1 if low_latency else self.settings["whisper_best_of"],
                    beam_size=1 if low_latency else self.settings["whisper_beam_size"],
                    patience=self.settings["whisper_patience"],
                    length_penalty=self.settings["whisper_length_penalty"],
                    suppress_tokens=self.settings["whisper_suppress_tokens"],
                    initial_prompt=self.settings["whisper_initial_prompt"] or None,
                    condition_on_previous_text=self.settings[
                        "whisper_condition_on_previous_text"
                    ],
                    temperature_increment_on_fallback=self.settings[
                        "whisper_temperature_increment_on_fallback"
                    ],
                    no_speech_threshold=self.settings["whisper_no_speech_threshold"],
                    logprob_threshold=self.settings["whisper_logprob_threshold"],
                    compression_ratio_threshold=self.settings[
                        "whisper_compression_ratio_threshold"
                    ],
                    endpoint_url=self.settings.get("whisper_endpoint_url"),
                    response_text_path=self.settings.get("whisper_response_text_path"),
                )
                rec = self.whisper_recognizer
                self.whisper_worker = LiveWhisperWorker(
                    transcribe_fn=lambda audio, interim: self._whisper_transcribe(
                        rec, audio, interim
                    ),
                    on_final=self._whisper_final,
                    on_interim=self._whisper_interim,
                    logger=self.logger,
                )
                self._last_interim_at = 0.0
                self.recognizer = None
                self.model = None
                self.moonshine_recognizer = None
                self.logger.log(
                    "Whisper live mode: "
                    + ("greedy decoding" if low_latency else "beam search")
                    + (", interim captions on" if self.settings.get("whisper_interim", True) else "")
                    + f", max segment {self._max_segment_ms() / 1000:.0f}s",
                    level="info",
                )

            elif engine == "moonshine":
                if not _MOONSHINE_AVAILABLE:
                    msg = "❌ Moonshine is not installed. Run: pip install moonshine-voice"
                    self.logger.log(msg, level="error")
                    return msg
                self.moonshine_recognizer = MoonshineRecognizer(
                    language=self.settings["moonshine_language"],
                    cache_dir=self.settings["moonshine_cache_dir"],
                    on_result=self._moonshine_result,
                    logger=self.logger,
                )
                self.moonshine_recognizer.start()

            # Stop mic monitor if it was running — recognition provides the level now
            if self.is_monitoring:
                self.stop_mic_monitor()

            # Reset/recreate VAD with current settings
            if self.vad:
                self.vad.flush()
            self.vad = None
            self._get_or_create_vad()

            # Reset subtitle buffer for new session
            self.subtitles.clear()

            # Start from live audio only — never from anything queued before.
            self._drain(self.audio_queue)

            self.stop_reason = None
            self._started_at = time.time()
            # Hardware: the watchdog expects callbacks from now on. Browser:
            # 0 = "no audio yet", which gets a longer grace period for the
            # permission prompt / screen-share picker.
            self._last_audio_at = (
                self._started_at if self.settings["audio_mode"] == "hardware" else 0.0
            )
            self.is_running = True
            self.state_seq += 1

            if (
                self.settings["audio_mode"] == "hardware"
                and microphone_index is not None
            ):
                self.settings["microphone"] = microphone_index
                device_info = sd.query_devices(microphone_index, "input")
                self.logger.log(
                    f"Using hardware mic: {device_info['name']}", level="info"
                )
                # 16 kHz if the device supports it, else its own rate
                # resampled to 16 kHz (see audio_input.py). 30 ms blocks.
                self.stream = open_input_stream(
                    microphone_index, self.audio_callback, logger=self.logger
                )
                self.stream.start()
                msg = "✅ Recognition started (Hardware)"
            else:
                msg = "✅ Recognition started (Browser)"

            self.process_thread = threading.Thread(
                target=self.process_audio_hardware, daemon=True
            )
            self.process_thread.start()

            if self.settings.get("translation_mode") == "ai":
                self.translation_service = TranslationService(
                    self.settings, self.logger
                )

            self.logger.log(msg, level="success")
            return msg

        except Exception as exc:
            msg = f"❌ Error starting recognition: {exc}"
            self.logger.log(msg, level="error")
            # Don't leave a half-started session "running" with no audio
            # source (e.g. the selected device no longer exists). The control
            # lock is an RLock, so this nested acquire is fine.
            if self.is_running:
                self.stop_recognition(reason=msg)
            return msg
        finally:
            self._control_lock.release()

    @staticmethod
    def _drain(q: queue.Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def check_audio_watchdog(self):
        """
        Stop the session if its audio source has gone away. Called about once
        a second by the global watchdog thread.

        A session with no audio source used to keep "running" forever — a
        server device that was unplugged just stopped invoking the callback,
        and a browser whose mic was disconnected (or whose tab was closed)
        just stopped sending. Now:
        - Server device: PortAudio calls back every 30 ms even in total
          silence, so no callbacks for a few seconds (or the stream going
          inactive) means the device is gone.
        - Browser: the page streams continuously, silence included, and
          reconnects on its own after a network blip — so no audio for
          `audio_watchdog_seconds` means the source really is gone.
        """
        if not self.is_running:
            return
        try:
            timeout = float(self.settings.get("audio_watchdog_seconds", 8) or 0)
        except (TypeError, ValueError):
            timeout = 8.0
        if timeout <= 0:
            return
        now = time.time()
        reason = None
        if self.settings.get("audio_mode") == "hardware":
            stream = self.stream
            if stream is not None and not stream.active:
                reason = "🔌 Audio device stopped (disconnected?) — session stopped"
            elif now - self._last_audio_at > min(timeout, _HW_WATCHDOG_MAX_S):
                reason = "🔌 No audio from the input device (disconnected?) — session stopped"
        else:
            if self._last_audio_at:
                silent_for = now - self._last_audio_at
                limit = timeout
            else:
                # Nothing received yet: allow for the mic permission prompt
                # or the screen-share picker.
                silent_for = now - self._started_at
                limit = max(timeout, _BROWSER_START_GRACE_S)
            if silent_for > limit:
                reason = (
                    f"🔌 No browser audio for {int(silent_for)}s (mic disconnected, "
                    f"sharing stopped, or tab closed) — session stopped"
                )
        if reason:
            self.stop_recognition(reason=reason)  # logs the reason

    def stop_recognition(self, reason: str | None = None) -> str:
        """
        Stop audio capture and unload all recognition models to free memory.
        Settings are fully preserved – pressing Start again will reload models.

        `reason` is shown in every open tab of this session; None means a
        normal user Stop.
        """
        # Blocking with a generous timeout: if Start is mid-way through loading
        # a big model, Stop should wait for that to finish and then unload it
        # cleanly, rather than race it. Better a brief wait than a crash.
        if not self._control_lock.acquire(timeout=30.0):
            return "⏳ Still starting — press Stop again in a moment"
        try:
            if not self.is_running and self.stream is None and self.recognizer is None:
                return "Already stopped"
            self.is_running = False
            self.stop_reason = reason
            self.state_seq += 1
            self._notify_audio_client(
                {"type": "stopped", "reason": reason or "⏹️ Recognition stopped"}
            )

            # Stop hardware stream. abort() rather than stop(): stop() waits
            # for buffers to drain, which can hang on a device that has just
            # been unplugged.
            if self.stream:
                try:
                    self.stream.abort()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            # Give the processing thread a moment to see is_running=False and
            # exit its loop before we start deleting the objects it's using.
            pt = getattr(self, "process_thread", None)
            if pt and pt.is_alive() and pt is not threading.current_thread():
                pt.join(timeout=1.0)

            # Clear audio buffers
            self.vosk_audio_buffer.clear()
            self._drain(self.audio_queue)

            # Flush any speech the VAD was accumulating mid-utterance (Whisper only)
            if self.vad:
                leftover = self.vad.flush()
                if leftover and self.whisper_worker:
                    self.whisper_worker.submit_final(leftover)
                self.vad = None

            # Let the worker finish the last utterance (in the background, so
            # Stop returns immediately), then release it.
            if self.whisper_worker:
                threading.Thread(
                    target=self.whisper_worker.close, daemon=True
                ).start()
                self.whisper_worker = None

            # Unload Vosk model to free memory
            self._unload_vosk()

            # Unload Whisper
            self.whisper_recognizer = None

            # Stop Moonshine — its Transcriber.stop() fires on_line_completed for any open line
            if self.moonshine_recognizer:
                self.moonshine_recognizer.close()
                self.moonshine_recognizer = None

            self.translation_service = None
            self._whisper_translator = None

            gc.collect()

            msg = reason or "⏹️ Recognition stopped (models unloaded)"
            self.logger.log(msg, level="warning" if reason else "success")
            return msg
        finally:
            self._control_lock.release()

    def _unload_vosk(self):
        """
        Release this session's KaldiRecognizer (always exclusive to this
        session — cheap) and its reference to the shared Vosk Model (only
        actually unloaded from RAM once every session using that same model
        path has stopped — see _release_vosk_model).
        """
        if self.recognizer:
            try:
                del self.recognizer
            except Exception:
                pass
            self.recognizer = None
        if self.model:
            _release_vosk_model(self._vosk_model_path)
            self.model = None
            self._vosk_model_path = None
        for _ in range(3):
            gc.collect()

    # ── Display ───────────────────────────────────────────────────────────────
    def _update_display_loop(self):
        while self.display_running:
            try:
                item = self.result_queue.get(timeout=0.02)  # 20ms max wait
                result_type, text = item[0], item[1]
                if result_type == "stop":
                    break
                if result_type == "final":
                    audio = item[2] if len(item) > 2 else self.last_audio_chunk
                    if not self.settings["enable_translation"]:
                        self.subtitles.add(text, "")
                    elif self.subtitles.mode == "instant":
                        # Show the recognized text now; the translation is
                        # filled in when it's ready (_translate_loop).
                        self.subtitles.add(text, "")
                        self._translate_queue.put((text, audio, "instant"))
                    else:
                        # Buffered mode pairs recognized+translated as one
                        # chunk, so it has to wait for the translation.
                        self._translate_queue.put((text, audio, "buffered"))
                elif result_type == "interim" and (
                    self.settings.get("display_interim")
                    or self.settings.get("recognition_engine") == "whisper"
                ):
                    self.subtitles.set_interim(text)
            except queue.Empty:
                if not self.display_running:
                    break
                time.sleep(0.005)
                # No sleep — tight loop so results appear as fast as possible.
                # CPU cost is negligible (most iterations are the 20ms queue timeout).
            except Exception as exc:
                if self.display_running:
                    self.logger.log(f"Display loop error: {exc}", level="error")

    def _translate_loop(self):
        """Translate finals off the display thread, in order."""
        while self.display_running:
            try:
                item = self._translate_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            text, audio, mode = item
            if mode == "instant":
                # Only the newest line is on screen in instant mode — if the
                # translator fell behind, skip straight to the latest line
                # instead of translating text nobody will see.
                while True:
                    try:
                        newer = self._translate_queue.get_nowait()
                    except queue.Empty:
                        break
                    if newer is None:
                        return
                    if newer[2] != "instant":
                        self._translate_queue.put(newer)
                        break
                    text, audio, mode = newer
            try:
                translated = self._translate(text, audio)
            except Exception as exc:
                self.logger.log(f"Translation error: {exc}", level="error")
                translated = ""
            if mode == "instant":
                self.subtitles.set_translation(text, translated)
            else:
                self.subtitles.add(text, translated)

    def _get_whisper_translator(self) -> WhisperRecognizer:
        # Reused between lines so its HTTP connection stays open (a new
        # client per line meant a new TCP/TLS handshake every time).
        if self._whisper_translator is None:
            self._whisper_translator = WhisperRecognizer(
                host=self.settings.get("whisper_translate_host"),
                api_key=self.settings.get("whisper_translate_api_key") or None,
                model=self.settings.get("whisper_translate_model"),
                logger=self.logger,
                temperature=self.settings["whisper_translate_temperature"],
                best_of=self.settings["whisper_translate_best_of"],
                beam_size=self.settings["whisper_translate_beam_size"],
                patience=self.settings["whisper_translate_patience"],
                length_penalty=self.settings["whisper_translate_length_penalty"],
                suppress_tokens=self.settings["whisper_translate_suppress_tokens"],
                initial_prompt=self.settings["whisper_translate_initial_prompt"]
                or None,
                condition_on_previous_text=self.settings[
                    "whisper_translate_condition_on_previous_text"
                ],
                temperature_increment_on_fallback=self.settings[
                    "whisper_translate_temperature_increment_on_fallback"
                ],
                no_speech_threshold=self.settings[
                    "whisper_translate_no_speech_threshold"
                ],
                logprob_threshold=self.settings[
                    "whisper_translate_logprob_threshold"
                ],
                compression_ratio_threshold=self.settings[
                    "whisper_translate_compression_ratio_threshold"
                ],
                endpoint_url=self.settings.get("whisper_translate_endpoint_url"),
                response_text_path=self.settings.get(
                    "whisper_translate_response_text_path"
                ),
            )
        return self._whisper_translator

    def _translate(self, text: str, audio: bytes | None = None) -> str:
        """Translate text using the configured translation mode."""
        mode = self.settings.get("translation_mode")
        audio = audio or self.last_audio_chunk
        try:
            if mode == "whisper" and audio:
                return self._get_whisper_translator().translate(audio)
            elif mode == "argos" and self.argos_translator:
                return self.argos_translator.translate(
                    text,
                    self.settings.get("argos_source_lang"),
                    self.settings.get("argos_target_lang"),
                )
            elif mode == "ai" and self.translation_service:
                return self.translation_service.translate(
                    text,
                    self.settings["source_language"].split("-")[0],
                    self.settings["target_language"],
                )
            elif mode == "libretranslate":
                # Create a temporary translation service for libretranslate
                ts = TranslationService(self.settings, self.logger)
                return ts.translate(
                    text,
                    self.settings["source_language"].split("-")[0],
                    self.settings["target_language"],
                )
        except Exception as exc:
            self.logger.log(f"Translation error: {exc}", level="error")
        return ""

    def get_display_html(self, recognized_text: str, translated_text: str) -> str:
        alignment_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
        base_style = f"font-family:{self._get_font_family_css()};white-space:pre-wrap;"
        rec_outline = self._get_outline_css(
            self.settings.get("outline_width", 0),
            self.settings.get("outline_color", "#000000"),
        )
        trans_outline = self._get_outline_css(
            self.settings.get("translated_outline_width", 0),
            self.settings.get("translated_outline_color", "#000000"),
        )

        def line_div(text, size, color, outline):
            return (
                f'<div style="font-size:{size}px;color:{color};{base_style}{outline}">'
                f"{html.escape(text)}</div>"
            )

        parts = []
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "before"
            and translated_text
        ):
            parts.append(
                line_div(
                    translated_text,
                    self.settings["translated_font_size"],
                    self.settings["translated_color"],
                    trans_outline,
                )
            )
        if recognized_text:
            parts.append(
                line_div(
                    recognized_text,
                    self.settings["recognized_font_size"],
                    self.settings["recognized_color"],
                    rec_outline,
                )
            )
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "after"
            and translated_text
        ):
            parts.append(
                line_div(
                    translated_text,
                    self.settings["translated_font_size"],
                    self.settings["translated_color"],
                    trans_outline,
                )
            )

        # The frame (background) always stays; only the .vt-lines block
        # fades. When the text goes empty the page's JS fades the block that
        # is already on screen instead of swapping in an empty one, so the
        # text and its outline disappear together (see vtRender).
        font_face = self._get_font_face_css()
        valign = self._VERTICAL_MAP.get(
            self.settings.get("vertical_alignment", "middle"), "center"
        )
        return (
            f"<style>{font_face}</style>"
            f'<div style="display:flex;flex-direction:column;justify-content:{valign};'
            f'padding:20px;background-color:{self.settings["background_color"]};min-height:200px;">'
            f'<div class="vt-lines" style="transition:opacity 0.5s;opacity:1;display:flex;'
            f"flex-direction:column;align-items:{alignment_map[self.settings['text_alignment']]};"
            f"text-align:{self.settings['text_alignment']};\">"
            + "".join(parts)
            + "</div></div>"
        )

    def get_current_display(self):
        rec, trans = self.subtitles.get_display()
        if not self.settings["enable_translation"]:
            trans = ""
        return (
            self.get_display_html(rec, trans),
            rec,
            trans,
        )

    def update_logs(self):
        return self.logger.get_recent_logs(50)

    # ── Display HTML ──────────────────────────────────────────────────────────
    def _get_font_family_css(self) -> str:
        if self.settings.get("custom_font"):
            font_name = Path(self.settings["custom_font"]).stem.replace(" ", "_")
            return f"'{font_name}', {self.settings['font_family']}"
        return self.settings["font_family"]

    def _get_font_face_css(self) -> str:
        custom_font = self.settings.get("custom_font")
        if not custom_font:
            return ""
        font_name = Path(custom_font).stem.replace(" ", "_")
        return (
            f"@font-face {{font-family:'{font_name}';"
            f"src:url('/fonts/{custom_font}') format('truetype');"
            f"font-weight:normal;font-style:normal;}}"
        )

    def _get_outline_css(self, width: int, color: str) -> str:
        if width == 0:
            return ""
        offsets = [
            f"{dx}px {dy}px 0 {color}"
            for dx in range(-width, width + 1)
            for dy in range(-width, width + 1)
            if not (dx == 0 and dy == 0)
        ]
        return "text-shadow: " + ", ".join(offsets) + ";"

    def get_display(self) -> tuple[str, str]:
        """Return (recognized, translated) for the current render frame."""
        with self._lock:
            now = time.time()

            if self.mode == "instant":
                if self._last_add == 0.0:
                    return "", ""
                if self._cur_rec or self._cur_trans:
                    if now - self._last_add <= self.fade_timeout:
                        rec = self._cur_rec if self._cur_rec else self._last_final_rec
                        trans = (
                            self._cur_trans
                            if self._cur_trans
                            else self._last_final_trans
                        )
                        return rec, trans
                return "", ""

            # ========== BUFFERED MODE ==========
            # First, check if current text's hold time has expired
            if self._cur_rec or self._cur_trans:
                if now >= self._show_until:
                    # Current text has expired - clear it
                    self._cur_rec = ""
                    self._cur_trans = ""
                    # If no more queued sentences, return empty
                    if not self._rec_queue:
                        return "", ""

            # If we have queued sentences, show the next chunk (only if nothing is currently showing)
            if self._rec_queue and not self._cur_rec and not self._cur_trans:
                rec_lines, trans_lines = [], []
                for _ in range(min(self.max_lines, len(self._rec_queue))):
                    rec_lines.append(self._rec_queue.pop(0))
                    if self._trans_queue:
                        trans_lines.append(self._trans_queue.pop(0))

                self._cur_rec = ", ".join(rec_lines)
                self._cur_trans = ", ".join(trans_lines)
                text_length = len(self._cur_rec)
                hold = max(text_length / self.cps, 2.5)  # Minimum 2.5 seconds
                self._show_until = now + hold
                return self._cur_rec, self._cur_trans

            # Return current text if it exists (and not expired, because expired would have been cleared above)
            if self._cur_rec or self._cur_trans:
                return self._cur_rec, self._cur_trans

            return "", ""

    _VERTICAL_MAP = {"top": "flex-start", "middle": "center", "bottom": "flex-end"}

    def popout_style_key(self) -> str:
        """Changes whenever anything baked into the popout page changes, so an
        open popout (e.g. an OBS browser source) reloads itself to pick up new
        styling instead of needing a manual refresh."""
        return hashlib.md5(self.generate_popout_html().encode()).hexdigest()[:12]

    def generate_popout_html(self) -> str:
        halign = self.settings.get("text_alignment", "center")
        valign = self._VERTICAL_MAP.get(
            self.settings.get("vertical_alignment", "middle"), "center"
        )
        font_face = self._get_font_face_css()
        font_family = self._get_font_family_css()
        rec_outline = self._get_outline_css(
            self.settings.get("outline_width", 0),
            self.settings.get("outline_color", "#000000"),
        )
        trans_outline = self._get_outline_css(
            self.settings.get("translated_outline_width", 0),
            self.settings.get("translated_outline_color", "#000000"),
        )
        # Fading is driven by the server (the text comes back empty once the
        # fade timeout / buffered hold is over). When that happens the page
        # keeps the old text on screen and fades the whole block — text,
        # outline and spacing together — and only empties it once the fade
        # has finished. It used to blank the text instantly and then fade an
        # empty container separately, which looked like a leftover box.
        # Empty lines are hidden so they don't leave a gap either.
        return (
            f'<!DOCTYPE html><html><head><title>Display</title><meta charset="UTF-8">'
            f"<style>{font_face}"
            f"body,html{{margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;"
            f"background:{self.settings['background_color']};}}"
            f"body{{display:flex;flex-direction:column;justify-content:{valign};}}"
            f".container{{box-sizing:border-box;padding:20px;width:100%;"
            f"text-align:{halign};transition:opacity 0.5s;opacity:1}}"
            f".container.fade{{opacity:0}}"
            f".rec,.tra{{margin:10px 0;font-family:{font_family};white-space:pre-wrap}}"
            f".rec:empty,.tra:empty{{display:none}}"
            f".rec{{font-size:{self.settings['recognized_font_size']}px;"
            f"color:{self.settings['recognized_color']};{rec_outline}}}"
            f".tra{{font-size:{self.settings['translated_font_size']}px;"
            f"color:{self.settings['translated_color']};{trans_outline}}}"
            f"</style>"
            f"<script>"
            f"let key=null;"
            f'const $=id=>document.getElementById(id);'
            f"async function update(){{try{{"
            f'const r=await fetch("/popout_data/{self.popout_id}",{{cache:"no-store"}});'
            f"if(!r.ok)return;const d=await r.json();"
            f"if(d.style_key){{if(key===null)key=d.style_key;"
            f"else if(d.style_key!==key){{location.reload();return}}}}"
            f'const rec=d.recognized||"",tra=d.translated||"";'
            f'const c=$("c");'
            f"if(rec||tra){{"
            f'if($("r").textContent!==rec)$("r").textContent=rec;'
            f'if($("t").textContent!==tra)$("t").textContent=tra;'
            f'c.classList.remove("fade")'
            f'}}else c.classList.add("fade")'
            f"}}catch(e){{}}}}"
            f'document.addEventListener("DOMContentLoaded",()=>{{'
            f'$("c").addEventListener("transitionend",()=>{{'
            f'if($("c").classList.contains("fade")){{$("r").textContent="";$("t").textContent=""}}}});'
            f"update();setInterval(update,150)}});"
            f"</script>"
            f"</head><body>"
            f'<div id="c" class="container fade">'
            + (
                '<div id="t" class="tra"></div><div id="r" class="rec"></div>'
                if self.settings.get("translation_position") == "before"
                else '<div id="r" class="rec"></div><div id="t" class="tra"></div>'
            )
            + "</div></body></html>"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def close(self):
        """Full cleanup: stop recognition, stop monitor, stop display thread."""
        if self.is_running:
            self.stop_recognition()
        if self.is_monitoring:
            self.stop_mic_monitor()

        self.display_running = False
        self._translate_queue.put(None)
        if self.display_thread and self.display_thread.is_alive():
            try:
                self.result_queue.put(("stop", ""))
                self.display_thread.join(timeout=2.0)
            except Exception:
                pass

        # Drain queues
        for q in (self.result_queue, self.audio_queue):
            self._drain(q)

        self.argos_translator = None
        gc.collect()
        print(f"[SESSION CLOSED] {self.slug}")

    def deactivate_session(self):
        self.session_active = False
        self.close()

    def touch(self):
        """Mark this session as recently active (used by the idle reaper)."""
        self.last_active = time.time()


# ── Global helpers ────────────────────────────────────────────────────────────
def get_available_models() -> list[tuple[str, str]]:
    """
    Scan vosk_models/ for anything that actually looks like a real Vosk
    model directory — not just any subdirectory. Every genuine Vosk model
    (small, big, or lgraph variant) contains at least one of a handful of
    well-known subfolders; without this check, a stray "temp" download
    folder, a partial/failed extraction, or an unrelated folder someone
    drops in there would show up as a selectable model and crash Vosk's
    Model() constructor the moment anyone actually picked it.
    """
    models_dir = Path("vosk_models")
    models_dir.mkdir(exist_ok=True)
    model_markers = ("am", "conf", "graph", "ivector", "rescore")
    results = []
    for item in sorted(models_dir.iterdir()):
        if not item.is_dir() or item.name.startswith(".") or item.name == "temp":
            continue
        if any((item / marker).exists() for marker in model_markers):
            results.append((item.name, str(item)))
    return results


def get_microphones() -> list[tuple[str, int]]:
    try:
        devices = sd.query_devices()
        return [
            (f"[ID:{i}] {d['name']}", i)
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception:
        return [("Error getting devices", 0)]


def _migrate_vad_threshold(v) -> float:
    """
    Migrate old 0–1 threshold values to dB.
    New slider range is -60 to 0 dB.  Old values were 0.0–1.0.
    A saved value > 0 and <= 1.0 is almost certainly the old format; convert it.
    Values already in dB are negative (or 0), so we can detect them trivially.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -30.0
    if v > 0:
        # Old 0–1 value: convert via rms formula then to dB
        # rms = 0.001 + v^1.5 * 0.499  →  dB = 20*log10(rms)
        import math

        rms = 0.001 + (v**1.5) * 0.499
        return max(-60.0, min(0.0, 20.0 * math.log10(max(rms, 1e-9))))
    # Already a negative dB value — clamp to valid range
    return max(-60.0, min(0.0, v))


# ── Shared Vosk model cache ───────────────────────────────────────────────────
# Vosk's own API is explicitly designed for this: one Model (the large,
# memory-heavy object — hundreds of MB to a few GB depending on which model)
# can back many independent KaldiRecognizer instances at once, safely, across
# threads. Previously every session called Model(model_path) itself, so two
# sessions using the *same* model path loaded it into RAM twice, three
# sessions loaded it three times, etc. This cache loads each distinct model
# path once and hands out a refcounted reference; the Model is only actually
# freed once every session using it has stopped. Whisper (remote API) and
# Moonshine (small ONNX models with per-session streaming state that can't be
# shared) aren't included — see RECOGNITION_QUALITY.md / MODEL_SHARING.md.
_VOSK_MODEL_LOCK = threading.Lock()
_VOSK_MODEL_CACHE: dict[str, dict] = {}  # model_path -> {"model": Model, "refcount": int}


def _acquire_vosk_model(model_path: str) -> Model:
    with _VOSK_MODEL_LOCK:
        entry = _VOSK_MODEL_CACHE.get(model_path)
        if entry is None:
            print(f"[VOSK] Loading model '{model_path}' (first session to use it)")
            entry = {"model": Model(model_path), "refcount": 0}
            _VOSK_MODEL_CACHE[model_path] = entry
        entry["refcount"] += 1
        print(
            f"[VOSK] '{model_path}' now shared by {entry['refcount']} session(s)"
        )
        return entry["model"]


def _release_vosk_model(model_path: str | None):
    if not model_path:
        return
    with _VOSK_MODEL_LOCK:
        entry = _VOSK_MODEL_CACHE.get(model_path)
        if entry is None:
            return
        entry["refcount"] -= 1
        if entry["refcount"] <= 0:
            print(f"[VOSK] Unloading '{model_path}' — no sessions using it anymore")
            del _VOSK_MODEL_CACHE[model_path]
            # The cache's reference to entry["model"] is gone; once this
            # session's own self.model reference is cleared too (right after
            # this call, in _unload_vosk), nothing references the underlying
            # C++ object anymore and it's freed on the next gc.collect().
        else:
            print(
                f"[VOSK] '{model_path}' still shared by {entry['refcount']} "
                f"other session(s) — not unloading"
            )


def _source_ended_reason(detail: str) -> str:
    detail = re.sub(r"[^\w .,:'()-]", "", str(detail or ""))[:80]
    return "🔌 Browser audio source ended" + (
        f" ({detail})" if detail else ""
    ) + " — session stopped"


def get_or_create_app(slug: str) -> VoiceTranslatorApp:
    with SESSION_LOCK:
        if slug not in SESSION_APPS:
            SESSION_APPS[slug] = VoiceTranslatorApp(slug)
            print(f"[NEW SESSION] {slug} | Total: {len(SESSION_APPS)}")
        app = SESSION_APPS[slug]
    app.touch()
    return app


def close_session(slug: str):
    with SESSION_LOCK:
        app = SESSION_APPS.pop(slug, None)
    if app:
        app.close()


# ── Idle session reaper (OFF by default) ────────────────────────────────────
# Sessions are permanent: nothing in this app ever tears down a session
# automatically. is_running/is_monitoring sessions are never touched here no
# matter how long they run or how much silence there is — silence is handled
# by the VAD producing no segments, not by touching the session. This reaper
# only ever considers sessions that are STOPPED (recognition off) and, even
# then, only if you explicitly opt in via IDLE_SESSION_TIMEOUT_SECONDS — set
# to 0 (default), it's fully disabled. Set a value only if you specifically
# want stopped-and-abandoned sessions to free their in-memory app object
# after N idle seconds (their settings.json is untouched either way).
IDLE_SESSION_TIMEOUT_SECONDS = int(os.environ.get("IDLE_SESSION_TIMEOUT_SECONDS", "0"))
_IDLE_REAPER_INTERVAL_SECONDS = 300


def _idle_reaper_loop():
    if IDLE_SESSION_TIMEOUT_SECONDS <= 0:
        return  # disabled — sessions live forever until explicitly closed
    while True:
        time.sleep(_IDLE_REAPER_INTERVAL_SECONDS)
        now = time.time()
        stale = []
        with SESSION_LOCK:
            for slug, app in SESSION_APPS.items():
                if app.is_running or app.is_monitoring:
                    continue  # never reap an actively-listening session
                if now - app.last_active > IDLE_SESSION_TIMEOUT_SECONDS:
                    stale.append(slug)
        for slug in stale:
            print(
                f"[IDLE REAPER] Closing '{slug}' after "
                f"{IDLE_SESSION_TIMEOUT_SECONDS}s stopped + idle"
            )
            close_session(slug)


threading.Thread(target=_idle_reaper_loop, daemon=True).start()


# ── Audio watchdog ────────────────────────────────────────────────────────────
# Stops any running session whose audio source has disappeared (unplugged
# device, disconnected browser mic, closed tab). Per-session opt-out via the
# "Auto-stop when audio is lost" setting (0 = never). See
# VoiceTranslatorApp.check_audio_watchdog.
def _audio_watchdog_loop():
    while True:
        time.sleep(1.0)
        with SESSION_LOCK:
            apps = list(SESSION_APPS.values())
        for app in apps:
            try:
                app.check_audio_watchdog()
            except Exception as exc:
                print(f"[WATCHDOG] {app.slug}: {exc}")


threading.Thread(target=_audio_watchdog_loop, daemon=True).start()


def find_app_by_popout_id(popout_id: str) -> "VoiceTranslatorApp | None":
    """
    The session that owns `popout_id`: a loaded one, or — straight after a
    server restart, before anyone reopened its UI — a saved one, which gets
    loaded. That's what lets an OBS browser source pointed at /popout/<id>
    keep working across restarts.
    """
    with SESSION_LOCK:
        for app in SESSION_APPS.values():
            if app.popout_id == popout_id:
                return app
    slug = find_slug_by_popout_id(popout_id)
    if slug:
        app = get_or_create_app(slug)
        if app.popout_id == popout_id:
            return app
    return None


def popout_id_in_use(popout_id: str, except_slug: str) -> bool:
    with SESSION_LOCK:
        for slug, app in SESSION_APPS.items():
            if slug != except_slug and app.popout_id == popout_id:
                return True
    owner = find_slug_by_popout_id(popout_id)
    return owner is not None and owner != except_slug


# ── UI ────────────────────────────────────────────────────────────────────────
def create_ui(args):  # noqa: C901  (complex but intentional)
    with gr.Blocks(title="Voice Translator", analytics_enabled=False) as interface:
        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                session_info = gr.Markdown("### Session initializing...")
                gr.Markdown("# 🎤 Voice Translator")
                browser_session_data = gr.HTML(visible=True)
                # Hidden: populated client-side (window.location / sessionStorage
                # — genuinely per-tab, unlike cookies) before handle_ui_load runs.
                # See the JS in the interface.load() chain near the bottom of
                # this function, and session.py's module docstring for why.
                slug_state = gr.Textbox(visible=False, elem_id="slug-state")
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(
                        "Visiting this site with no name gives you the **main** "
                        "session, always the same one. Open or create a separate, "
                        "independently-configured session below.",
                        elem_id="session-switcher-hint",
                    )
                    with gr.Row():
                        new_session_name = gr.Textbox(
                            label="Open or create a session by name",
                            placeholder="e.g. khyretos",
                            scale=3,
                        )
                        open_session_btn = gr.Button("↗️ Go", scale=1, size="sm")
                        random_session_btn = gr.Button(
                            "🎲 New", scale=1, size="sm",
                            elem_id="random-session-btn",
                        )
                    session_dropdown = gr.Dropdown(
                        label="Manage Sessions",
                        choices=[],
                        value=None,
                        interactive=True,
                        scale=4,
                        info="Select a session, then Close it — selecting alone no "
                        "longer closes anything.",
                    )
                    refresh_sessions_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                    close_session_btn = gr.Button(
                        "🗑️ Close Selected Session", size="sm", variant="stop"
                    )

        with gr.Row():
            # ── Left column: settings ─────────────────────────────────────────
            with gr.Column(scale=1):
                # Speech recognition
                with gr.Accordion("🗣️ Speech Recognition", open=True):
                    engine_choices = ["vosk", "whisper"]
                    if _MOONSHINE_AVAILABLE:
                        engine_choices.append("moonshine")
                    recognition_engine = gr.Radio(
                        engine_choices,
                        value="vosk",
                        label="Recognition Engine",
                        info="Vosk: offline/fast | Whisper: online/accurate | Moonshine: local/lightweight"
                        + (
                            ""
                            if _MOONSHINE_AVAILABLE
                            else " (Moonshine not installed, run `pip install moonshine-voice`)"
                        ),
                    )

                    with gr.Group(visible=True) as vosk_settings:
                        vosk_models = get_available_models()
                        vosk_choices = (
                            vosk_models
                            if vosk_models
                            else [("❌ No models detected", "")]
                        )
                        vosk_value = vosk_models[0][1] if vosk_models else ""
                        vosk_model_dropdown = gr.Dropdown(
                            choices=vosk_choices,
                            value=vosk_value,
                            label="Vosk Model",
                            info=f"Found {len(vosk_models)} models in vosk_models/"
                            if vosk_models
                            else "⚠️ No models found",
                            interactive=bool(vosk_models),
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
                        display_interim = gr.Checkbox(
                            label="Show interim results",
                            value=True,
                            info="Display partial recognition while speaking (Vosk only)",
                        )

                    with gr.Group(visible=False) as whisper_settings:
                        whisper_host = gr.Textbox(
                            label="Whisper API Host", value="http://localhost:9000"
                        )
                        whisper_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Leave empty if not required",
                        )
                        whisper_model = gr.Textbox(
                            label="Model", value="whisper-large-v3"
                        )
                        whisper_language = gr.Textbox(label="Language", value="en")
                        with gr.Group():
                            gr.Markdown(
                                "**Live mode** · for conversation: captions appear "
                                "while you speak instead of after you stop"
                            )
                            whisper_low_latency = gr.Checkbox(
                                value=True,
                                label="Low-latency decoding",
                                info="Greedy decoding (beam size 1, best-of 1) — overrides "
                                "the Beam size / Best of sliders below. Several times faster "
                                "per request on most servers for a small accuracy cost.",
                            )
                            whisper_interim = gr.Checkbox(
                                value=True,
                                label="Live partial captions",
                                info="Show a best guess of the sentence in progress, "
                                "refreshed about every 0.8 s while the server is idle. "
                                "Never delays the final text.",
                            )
                            whisper_max_segment_s = gr.Slider(
                                2,
                                30,
                                6,
                                step=1,
                                label="Max segment length (s)",
                                info="Continuous speech is sent in pieces of at most this "
                                "long (cut at the quietest point), instead of waiting for "
                                "you to pause. Lower = more live; higher = more context.",
                            )
                        with gr.Accordion("Advanced Whisper Parameters", open=False):
                            whisper_temperature = gr.Slider(
                                0.0, 1.0, 0.0, step=0.1, label="Temperature"
                            )
                            whisper_best_of = gr.Slider(
                                1, 10, 5, step=1, label="Best of"
                            )
                            whisper_beam_size = gr.Slider(
                                1, 10, 5, step=1, label="Beam size"
                            )
                            whisper_patience = gr.Slider(
                                0.0, 2.0, 1.0, step=0.1, label="Patience"
                            )
                            whisper_length_penalty = gr.Slider(
                                0.0, 2.0, 1.0, step=0.1, label="Length penalty"
                            )
                            whisper_suppress_tokens = gr.Textbox(
                                value="-1", label="Suppress tokens"
                            )
                            whisper_initial_prompt = gr.Textbox(
                                label="Initial prompt", placeholder="Optional"
                            )
                            whisper_condition_on_previous_text = gr.Checkbox(
                                value=False,
                                label="Condition on previous text",
                                info="Off by default — biases decoding toward whatever "
                                "the previous segment said, which can amplify hallucination "
                                "loops (once it says 'Thank you.' once, it's more likely to "
                                "keep saying outro-style filler on the next quiet segment).",
                            )
                            whisper_temperature_increment_on_fallback = gr.Slider(
                                0.0,
                                1.0,
                                0.2,
                                step=0.1,
                                label="Temperature increment on fallback",
                            )
                            whisper_no_speech_threshold = gr.Slider(
                                0.0, 1.0, 0.6, step=0.1, label="No speech threshold"
                            )
                            whisper_logprob_threshold = gr.Slider(
                                -5.0, 0.0, -1.0, step=0.1, label="Logprob threshold"
                            )
                            whisper_compression_ratio_threshold = gr.Slider(
                                0.0,
                                5.0,
                                2.4,
                                step=0.1,
                                label="Compression ratio threshold",
                            )
                            with gr.Accordion(
                                "🛠️ Advanced: custom endpoint & response shape (developer)",
                                open=False,
                            ):
                                gr.Markdown(
                                    "For a Whisper-compatible server that isn't shaped "
                                    "like OpenAI's API. Leave blank to use the default: "
                                    "`{API Host}/audio/transcriptions` and standard "
                                    "verbose_json response parsing. See AI_TRANSLATION.md."
                                )
                                whisper_endpoint_url = gr.Textbox(
                                    label="Full endpoint URL override",
                                    placeholder="e.g. https://api.example.com/v1/transcribe",
                                    info="Used exactly as-is instead of deriving from "
                                    "API Host + /audio/transcriptions.",
                                )
                                whisper_response_text_path = gr.Textbox(
                                    label="Response text path",
                                    placeholder="e.g. result.text (default: standard "
                                    "verbose_json segments — see AI_TRANSLATION.md)",
                                    info="Dot-path to the transcribed text in a "
                                    "non-standard JSON response. When set, this bypasses "
                                    "the no_speech/logprob/compression_ratio filtering "
                                    "above entirely, since a custom shape can't be "
                                    "assumed to have that segment data.",
                                )
                        test_whisper_btn = gr.Button("🔍 Test Connection", size="sm")

                    with gr.Group(visible=False) as moonshine_settings:
                        _moon_status = (
                            "✅ moonshine-voice installed"
                            if _MOONSHINE_AVAILABLE
                            else "⚠️ Install with: `pip install moonshine-voice`"
                        )
                        gr.Markdown(
                            f"{_moon_status}  \n"
                            "Model is **auto-downloaded** on first Start for the selected language."
                        )
                        moonshine_language = gr.Dropdown(
                            choices=MOONSHINE_LANGUAGES,
                            value="en",
                            label="Language",
                            info="Best available model for that language is downloaded automatically.",
                        )
                        moonshine_cache_dir = gr.Textbox(
                            label="Cache Directory",
                            value="moonshine_models",
                            info="Downloaded models are stored here (MOONSHINE_VOICE_CACHE).",
                        )

                # Audio
                with gr.Accordion("🎙️ Audio", open=True):
                    audio_mode = gr.Radio(
                        [
                            ("Server Audio Device", "hardware"),
                            ("Browser Microphone", "browser_mic"),
                            ("Browser Tab / System Audio", "browser_display"),
                        ],
                        value="hardware",
                        label="Audio Source",
                        info=(
                            "Server Device: any input on the host — including a virtual/"
                            "loopback device if you route another app's output through one "
                            "(see AUDIO_SOURCES.md, e.g. for Discord's desktop app) | "
                            "Browser Microphone: this device's mic | Browser Tab/System Audio: "
                            "share a tab or your screen with 'share audio' checked (e.g. "
                            "Discord in a browser tab, or Windows/ChromeOS system audio)"
                        ),
                    )

                    with gr.Group(visible=True) as hardware_group:
                        microphones = get_microphones()
                        mic_dropdown = gr.Dropdown(
                            choices=microphones,
                            label="Input Device",
                            value=microphones[0][1] if microphones else None,
                            interactive=True,
                            info=(
                                "Lists every input-capable device the server can see — "
                                "physical mics AND any virtual/loopback device configured "
                                "on the host (e.g. a PulseAudio monitor source or VB-Audio "
                                "Cable). See AUDIO_SOURCES.md to route another app's audio "
                                "into one of those."
                            ),
                        )
                        with gr.Row():
                            refresh_mic_btn = gr.Button("🔄 Refresh", size="sm")
                            test_mic_btn = gr.Button(
                                "🎙️ Test Mic", size="sm", variant="secondary"
                            )
                            stop_test_mic_btn = gr.Button(
                                "⏹ Stop Test", size="sm", visible=False
                            )
                        mic_test_status = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            lines=1,
                            visible=False,
                        )

                    with gr.Group(visible=False) as browser_group:
                        browser_source_hint = gr.Markdown(
                            "Uses this device's microphone.", visible=True
                        )
                        browser_status = gr.Textbox(
                            label="Browser Stream Status",
                            value="Not started",
                            interactive=False,
                            elem_id="browser-status",
                        )
                        with gr.Row():
                            browser_test_mic_btn = gr.Button(
                                "🎙️ Test Mic", size="sm", variant="secondary"
                            )
                            browser_stop_test_mic_btn = gr.Button(
                                "⏹ Stop Test", size="sm", visible=False
                            )

                    # ── VAD + Level meter (always visible, works in test mode too) ──
                    with gr.Group():
                        _vad_backend = (
                            "WebRTC VAD"
                            if _WRTCVAD_AVAILABLE
                            else "RMS energy (install webrtcvad for better accuracy)"
                        )
                        gr.Markdown(f"**Voice Detection** · {_vad_backend}")

                        vad_threshold = gr.Slider(
                            minimum=-60,
                            maximum=0,
                            value=-30,
                            step=1,
                            label="Detection threshold (dB)",
                            info="Set where the orange line sits on the meter above. "
                            "-60 = detects a whisper · -30 = normal speech · -10 = loud speech only.",
                            elem_id="vad-threshold-slider",
                        )
                        vad_end_silence_ms = gr.Slider(
                            minimum=0,
                            maximum=2000,
                            value=300,
                            step=50,
                            label="End-of-speech pause (ms)",
                            info="How long silence must last before a segment is sent to "
                            "Whisper/Moonshine. Default raised from 80ms → 300ms: 80ms is "
                            "shorter than a normal mid-sentence breath, so it was fragmenting "
                            "speech into tiny clips — and very short/ambiguous clips are "
                            "exactly what triggers Whisper to hallucinate fillers like "
                            "'thank you'. Raise further (500–800ms) if it's still cutting "
                            "off mid-sentence; lower it if replies feel laggy.",
                        )
                    vad_min_speech_ms = gr.Slider(
                        minimum=50,
                        maximum=800,
                        value=200,
                        step=10,
                        label="Ignore sounds shorter than (ms)",
                        info="Whisper only. A sound needs at least this much actual speech "
                        "to be sent — desk knocks, clicks and coughs fall below it, and "
                        "those are exactly what Whisper turns into 'Thank you' / "
                        "'Subscribe'. Raise if knocks still get through; lower if very "
                        "short replies ('yes', 'no') get dropped.",
                    )
                    audio_watchdog_seconds = gr.Slider(
                        minimum=0,
                        maximum=60,
                        value=8,
                        step=1,
                        label="Auto-stop when audio is lost (s)",
                        info="Stop the session automatically when its audio source "
                        "disappears: the server device is unplugged, or no audio has "
                        "arrived from the browser (mic disconnected, sharing stopped, "
                        "tab closed) for this many seconds. Brief network drops "
                        "reconnect on their own. 0 = never auto-stop.",
                    )
                    # After vad_end_silence_ms slider, add:
                    with gr.Group():
                        gr.Markdown(
                            "**Noise filter · clicks, keyboard, background hum**"
                        )
                        noise_filter_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0,
                            step=0.05,
                            label="Noise filter strength",
                            info="0 = off · 0.3 = light (removes obvious clicks) · 0.7 = medium "
                            "(spectral subtraction + transient suppression) · 1.0 = aggressive "
                            "(maximum cleaning, may soften quiet speech). Changes take effect immediately.",
                        )
                        # Level meter — shows EXACTLY what the VAD sees (no smoothing).
                        # Orange vertical line = detection threshold.
                        # White vertical line  = peak hold (2 s decay).
                        # Bar turns RED when above threshold (VAD is active).
                        gr.HTML("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
  <div id="vad-led" style="width: 14px; height: 14px; border-radius: 50%; background: #888; box-shadow: 0 0 2px #000; transition: background 0.05s;"></div>
  <span style="font-size: 12px; color: #ccc;">VAD active</span>
</div>
<div style="margin:4px 0 2px; font-size:11px; color:#aaa; display:flex; justify-content:space-between; user-select:none;">
  <span>-60</span><span>-45</span><span>-30</span><span>-15</span><span>0 dB</span>
</div>
<div style="position:relative; background:#1a1a1a; height:20px; border-radius:4px;
            overflow:visible; border:1px solid #444; margin-bottom:2px;">
  <!-- Live level bar (no smoothing — matches VAD exactly) -->
  <div id="volume-meter-bar"
       style="height:100%; width:0%; border-radius:3px; background:#44cc44; transition:none;">
  </div>
  <!-- Threshold line (orange) -->
  <div id="volume-meter-threshold-line"
       style="position:absolute; top:-5px; bottom:-5px; width:3px; background:#f5a623;
              border-radius:2px; left:50%; pointer-events:none; z-index:10;
              box-shadow:0 0 5px #f5a623;" title="Detection threshold">
  </div>
  <!-- Peak-hold line (white) — holds max level for 2 s -->
  <div id="volume-meter-peak-line"
       style="position:absolute; top:2px; bottom:2px; width:2px; background:#fff;
              border-radius:1px; left:0%; pointer-events:none; z-index:9;
              opacity:0.8;" title="Peak hold">
  </div>
</div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:#888; margin-top:1px;">
  <span id="volume-meter-db-label">— dB</span>
  <span style="color:#f5a623; font-weight:bold;" id="volume-meter-threshold-label">threshold: — dB</span>
</div>
<div style="font-size:10px; color:#666; margin-top:2px;">
  Bar = live level (no smoothing) · White line = peak hold (2 s) · Red bar = above threshold
</div>""")

                # Translation
                with gr.Accordion("🌐 Translation", open=True):
                    enable_translation = gr.Checkbox(
                        label="Enable Translation",
                        value=False,
                        info="Toggle translation on/off",
                    )
                    with gr.Group(visible=False) as translation_group:
                        translation_mode = gr.Radio(
                            ["argos", "ai", "libretranslate", "whisper"],
                            value="argos",
                            label="Translation Engine",
                        )
                        with gr.Row():
                            source_lang = gr.Textbox(
                                label="From", value="en", scale=1, visible=False
                            )
                            target_lang = gr.Textbox(
                                label="To", value="es", scale=1, visible=False
                            )

                        with gr.Group(visible=False) as ai_settings:
                            ai_host = gr.Textbox(
                                label="API Host",
                                value="http://localhost:11434",
                                info="Base host only — the /v1/chat/completions path is "
                                "added automatically. Was hardcoded to guess a "
                                "'/v3/...' path here before, which isn't a real endpoint "
                                "on any standard OpenAI-compatible server; fixed to /v1.",
                            )
                            ai_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="Leave empty for Ollama",
                            )
                            ai_model = gr.Textbox(label="Model", value="llama3.2")

                            with gr.Accordion(
                                "🛠️ Advanced: custom prompt & request shape (developer)",
                                open=False,
                            ):
                                gr.Markdown(
                                    "Full details and examples in **AI_TRANSLATION.md**. "
                                    "Leave any of these blank to use the sensible "
                                    "OpenAI/Ollama-compatible default."
                                )
                                ai_translation_prompt_template = gr.Textbox(
                                    label="Prompt template",
                                    lines=3,
                                    placeholder=tmod.DEFAULT_AI_PROMPT_TEMPLATE,
                                    info="Placeholders: {source_lang} {target_lang} {text}",
                                )
                                ai_endpoint_url = gr.Textbox(
                                    label="Full endpoint URL override",
                                    placeholder="e.g. https://api.example.com/v1/chat/completions",
                                    info="If set, used exactly as-is instead of deriving "
                                    "from API Host above — for APIs whose chat-completion "
                                    "path isn't /v1/chat/completions at all.",
                                )
                                ai_request_body_template = gr.Textbox(
                                    label="Request body template (JSON)",
                                    lines=8,
                                    placeholder=tmod.DEFAULT_AI_REQUEST_BODY_TEMPLATE,
                                    info="__MODEL__ and __PROMPT__ are substituted as "
                                    "JSON-escaped string content. Use this to match a "
                                    "non-OpenAI-compatible request shape entirely.",
                                )
                                ai_response_text_path = gr.Textbox(
                                    label="Response text path",
                                    placeholder=tmod.DEFAULT_AI_RESPONSE_TEXT_PATH,
                                    info="Dot-path to the translated text in the JSON "
                                    "response, e.g. 'choices.0.message.content' or "
                                    "'result.translation'. Numeric segments index lists.",
                                )
                                ai_advanced_reset_btn = gr.Button(
                                    "Reset these 4 fields to default", size="sm"
                                )

                        with gr.Group(visible=False) as libretranslate_settings:
                            libretranslate_host = gr.Textbox(
                                label="Host", value="http://localhost:5000"
                            )
                            libretranslate_api_key = gr.Textbox(
                                label="API Key", type="password"
                            )

                        with gr.Group(visible=False) as whisper_translate_settings:
                            whisper_trans_host = gr.Textbox(
                                label="Whisper API Host", value="http://localhost:9000"
                            )
                            whisper_trans_api_key = gr.Textbox(
                                label="API Key", type="password"
                            )
                            whisper_trans_model = gr.Textbox(
                                label="Model", value="whisper-large-v3"
                            )
                            with gr.Accordion(
                                "Advanced Whisper Translate Parameters", open=False
                            ):
                                whisper_trans_temperature = gr.Slider(
                                    0.0, 1.0, 0.0, step=0.1, label="Temperature"
                                )
                                whisper_trans_best_of = gr.Slider(
                                    1, 10, 5, step=1, label="Best of"
                                )
                                whisper_trans_beam_size = gr.Slider(
                                    1, 10, 5, step=1, label="Beam size"
                                )
                                whisper_trans_patience = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.1, label="Patience"
                                )
                                whisper_trans_length_penalty = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.1, label="Length penalty"
                                )
                                whisper_trans_suppress_tokens = gr.Textbox(
                                    value="-1", label="Suppress tokens"
                                )
                                whisper_trans_initial_prompt = gr.Textbox(
                                    label="Initial prompt", placeholder="Optional"
                                )
                                whisper_trans_condition_on_previous_text = gr.Checkbox(
                                    value=False,
                                    label="Condition on previous text",
                                    info="Off by default — see the same setting in "
                                    "Whisper recognition above for why.",
                                )
                                whisper_trans_temperature_increment_on_fallback = (
                                    gr.Slider(
                                        0.0,
                                        1.0,
                                        0.2,
                                        step=0.1,
                                        label="Temperature increment on fallback",
                                    )
                                )
                                whisper_trans_no_speech_threshold = gr.Slider(
                                    0.0, 1.0, 0.6, step=0.1, label="No speech threshold"
                                )
                                whisper_trans_logprob_threshold = gr.Slider(
                                    -5.0, 0.0, -1.0, step=0.1, label="Logprob threshold"
                                )
                                whisper_trans_compression_ratio_threshold = gr.Slider(
                                    0.0,
                                    5.0,
                                    2.4,
                                    step=0.1,
                                    label="Compression ratio threshold",
                                )
                                with gr.Accordion(
                                    "🛠️ Advanced: custom endpoint & response shape "
                                    "(developer)",
                                    open=False,
                                ):
                                    whisper_trans_endpoint_url = gr.Textbox(
                                        label="Full endpoint URL override",
                                        placeholder="e.g. https://api.example.com/v1/translate",
                                        info="Used exactly as-is instead of deriving "
                                        "from API Host + /audio/translations.",
                                    )
                                    whisper_trans_response_text_path = gr.Textbox(
                                        label="Response text path",
                                        placeholder="e.g. result.text",
                                        info="Same idea as the recognition-side setting "
                                        "— see AI_TRANSLATION.md.",
                                    )
                            gr.Markdown(
                                "*Translates audio directly to English using Whisper*"
                            )

                        with gr.Group(visible=True) as argos_settings:
                            argos_source = gr.Textbox(
                                label="Source Language Code",
                                value="en",
                                info="2-letter code (en, es, fr…)",
                            )
                            argos_target = gr.Textbox(
                                label="Target Language Code", value="es"
                            )
                            if not ARGOS_AVAILABLE:
                                gr.Markdown(
                                    "⚠️ *Install argostranslate: pip install argostranslate*"
                                )
                            else:
                                installed_pairs = (
                                    argostranslate.translate.get_installed_languages()
                                )
                                if len(installed_pairs) < 2:
                                    gr.Markdown(
                                        "⚠️ **No language pairs installed.** Run `download_argos_models.py`."
                                    )
                                else:
                                    gr.Markdown("✅ Argos is ready.")

            # ── Right column: display + style ─────────────────────────────────
            with gr.Column(scale=1):
                with gr.Row():
                    start_btn = gr.Button("▶️ Start", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop")

                status_text = gr.Textbox(label="Status", lines=2, elem_id="status-text")

                gr.Markdown("### 📺 Display")
                popout_url = gr.Textbox(label="Popout URL", interactive=False)
                with gr.Group():
                    gr.Markdown(
                        "For a custom ID type it and press Enter (or click away). "
                        "The ID is saved with this session, so the OBS URL keeps "
                        "working after restarts."
                    )
                    custom_popout_id = gr.Textbox(
                        label="Custom Popout ID",
                        placeholder="Enter custom ID or leave empty for random",
                        scale=3,
                    )
                    random_btn = gr.Button("🎲 Random", scale=1, size="sm")

                # Filled in by the page's JS polling (/display_data), not by Gradio.
                display_html = gr.HTML(
                    label="Display",
                    value='<div id="vt-display"><div>Loading...</div></div>',
                )

                with gr.Accordion("Outputs", open=False):
                    recognized_output = gr.Textbox(
                        label="Recognized",
                        elem_id="recognized-output-text",
                    )
                    translated_output = gr.Textbox(
                        label="Translated",
                        elem_id="translated-output-text",
                    )

                with gr.Accordion("🎨 Display Style", open=False):
                    custom_fonts = get_available_fonts()
                    font_choices = SYSTEM_FONTS + custom_fonts
                    font_selector = gr.Dropdown(
                        choices=font_choices,
                        value="Arial",
                        label="Font Family",
                        info="Select a system font or a custom font from fonts/",
                    )
                    refresh_fonts_btn = gr.Button("🔄 Refresh Fonts", size="sm")
                    recognized_font_size = gr.Slider(
                        12, 120, 48, step=2, label="Main text size"
                    )
                    translated_font_size = gr.Slider(
                        12, 120, 32, step=2, label="Translation size"
                    )
                    with gr.Row():
                        recognized_color = gr.ColorPicker(
                            label="Main color", value="#FFFFFF"
                        )
                        translated_color = gr.ColorPicker(
                            label="Translation color", value="#CCCCCC"
                        )
                        background_color = gr.ColorPicker(
                            label="Background", value="#000000"
                        )
                    with gr.Row():
                        text_alignment = gr.Radio(
                            ["left", "center", "right"], value="center", label="Align"
                        )
                        vertical_alignment = gr.Radio(
                            [("Top", "top"), ("Middle", "middle"), ("Bottom", "bottom")],
                            value="middle",
                            label="Vertical position",
                            info="Bottom + a popout the size of your stream = "
                            "classic subtitles",
                        )
                    translation_position = gr.Radio(
                        ["before", "after"], value="after", label="Translation position"
                    )
                    fade_timeout = gr.Slider(
                        0.1, 10.0, 5.0, step=0.1, label="Fade Timeout (seconds)"
                    )

                    with gr.Group():
                        gr.Markdown("**Recognized Text Outline**")
                        with gr.Row():
                            with gr.Column(scale=3):
                                outline_width = gr.Slider(
                                    0, 10, 0, step=1, label="Outline width (px)"
                                )
                            with gr.Column(scale=1):
                                outline_color = gr.ColorPicker(
                                    label="Outline color", value="#000000"
                                )

                    with gr.Group():
                        gr.Markdown("**Translated Text Outline**")
                        with gr.Row():
                            with gr.Column(scale=3):
                                translated_outline_width = gr.Slider(
                                    0, 10, 0, step=1, label="Outline width (px)"
                                )
                            with gr.Column(scale=1):
                                translated_outline_color = gr.ColorPicker(
                                    label="Outline color", value="#000000"
                                )

                with gr.Accordion("📺 Subtitle Timing", open=False):
                    gr.Markdown(
                        "**Instant** — subtitles appear immediately and fade after the timeout.  \n"
                        "**Buffered** — sentences queue up and are shown as chunks, "
                        "each held long enough to read at the chosen CPS speed. "
                        "Good for fast speakers or Whisper which sends whole phrases."
                    )
                    subtitle_mode = gr.Radio(
                        choices=[
                            ("Instant", "instant"),
                            ("Buffered / paced", "buffered"),
                        ],
                        value="instant",
                        label="Display mode",
                    )
                    with gr.Group(visible=False) as subtitle_buffered_group:
                        subtitle_cps = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=21,
                            step=1,
                            label="Reading speed (characters per second)",
                            info="Netflix standard ≈ 17 CPS for most languages. "
                            "Raise for faster readers, lower to give more time per chunk.",
                        )
                        subtitle_max_lines = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            label="Sentences per chunk",
                            info="How many sentences to show at once before advancing.",
                        )

                with gr.Row():
                    reset_defaults_btn = gr.Button(
                        "🔄 Reset to Defaults", variant="secondary", size="sm"
                    )

        log_output = gr.Textbox(label="Log", lines=6, elem_id="log_output")

        # ── Event handlers ────────────────────────────────────────────────────

        def _set(key: str, persist: bool = True):
            """Return a function that sets app.settings[key] = value (and persists)."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                if persist:
                    persist_settings(app.slug, app.settings)

            return updater

        def _set_vad(key: str):
            """Set VAD setting AND push to running VAD immediately."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                app.apply_vad_settings()
                persist_settings(app.slug, app.settings)

            return updater

        def _set_subtitle(key: str):
            """Set subtitle setting AND apply to SubtitleManager immediately."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                app.apply_subtitle_settings()
                persist_settings(app.slug, app.settings)

            return updater

        def _set_fade(value, request: gr.Request):
            """Fade timeout applies to both settings and subtitle manager."""
            app = get_or_create_app(get_slug(request))
            app.settings["fade_timeout"] = value
            app.apply_subtitle_settings()
            persist_settings(app.slug, app.settings)

        # Subtitle mode toggle
        def update_subtitle_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["subtitle_mode"] = mode
            app.apply_subtitle_settings()
            persist_settings(app.slug, app.settings)
            return gr.update(visible=(mode == "buffered"))

        # Mic test handlers
        def start_mic_test(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            msg = app.start_mic_monitor()
            return (
                msg,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        def stop_mic_test(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.stop_mic_monitor()
            return (
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        def start_browser_mic_test(request: gr.Request):
            return "", gr.update(visible=True), gr.update(visible=False)

        def stop_browser_mic_test(request: gr.Request):
            return "", gr.update(visible=False), gr.update(visible=True)

        def update_recognition_engine(engine, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["recognition_engine"] = engine
            persist_settings(app.slug, app.settings)
            return {
                vosk_settings: gr.update(visible=(engine == "vosk")),
                whisper_settings: gr.update(visible=(engine == "whisper")),
                moonshine_settings: gr.update(visible=(engine == "moonshine")),
            }

        def update_audio_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["audio_mode"] = mode
            persist_settings(app.slug, app.settings)
            if mode == "browser_display":
                hint = (
                    "**Tab/System Audio**: pressing Start (or Test Mic) will ask "
                    "your browser to pick a tab, window, or screen — check **'Share "
                    "audio'** in that picker. What actually gets captured (just that "
                    "tab vs. the whole system) depends on your browser/OS — see "
                    "AUDIO_SOURCES.md. This is how to translate e.g. a Discord *web* "
                    "tab, or (Windows/ChromeOS) whole-system audio."
                )
            else:
                hint = "Uses this device's microphone."
            return {
                hardware_group: gr.update(visible=(mode == "hardware")),
                browser_group: gr.update(
                    visible=(mode in ("browser_mic", "browser_display"))
                ),
                browser_source_hint: gr.update(value=hint),
            }

        def update_translation_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["translation_mode"] = mode
            persist_settings(app.slug, app.settings)
            return {
                ai_settings: gr.update(visible=(mode == "ai")),
                libretranslate_settings: gr.update(visible=(mode == "libretranslate")),
                whisper_translate_settings: gr.update(visible=(mode == "whisper")),
                argos_settings: gr.update(visible=(mode == "argos")),
                source_lang: gr.update(visible=(mode != "argos")),
                target_lang: gr.update(visible=(mode != "argos")),
            }

        def update_translation_toggle(enabled, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["enable_translation"] = enabled
            if not enabled:
                app.subtitles._cur_trans = ""
            persist_settings(app.slug, app.settings)
            return gr.update(visible=enabled)

        def update_font_selector(value, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            custom_fonts = [f[1] for f in get_available_fonts()]
            if value in custom_fonts:
                app.settings["custom_font"] = value
            else:
                app.settings["font_family"] = value
                app.settings["custom_font"] = ""
            persist_settings(app.slug, app.settings)

        def _popout_url(request: gr.Request, app: VoiceTranslatorApp) -> str:
            # Build the URL the browser actually used (host + scheme, incl.
            # behind a reverse proxy) instead of the bind address, which is
            # usually 0.0.0.0 and not something OBS can open.
            base = f"http://{args.host}:{args.port}"
            try:
                headers = request.headers
                host = headers.get("x-forwarded-host") or headers.get("host")
                if host:
                    proto = headers.get("x-forwarded-proto") or "http"
                    base = f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}"
            except Exception:
                pass
            return f"{base}/popout/{app.popout_id}"

        def update_custom_popout_id(value, request: gr.Request):
            # The id is saved with the session's settings (see
            # VoiceTranslatorApp.popout_id), so the OBS URL survives restarts.
            app = get_or_create_app(get_slug(request))
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", (value or "").strip())[:64]
            if not sanitized:
                app.popout_id = secrets.token_urlsafe(16)
                status = f"🎲 New random popout ID saved: {app.popout_id}"
            elif sanitized == app.popout_id:
                status = f"✅ Popout ID unchanged: {sanitized}"
            elif popout_id_in_use(sanitized, app.slug):
                status = (
                    f"❌ Popout ID '{sanitized}' is already used by another "
                    f"session — kept '{app.popout_id}'"
                )
            else:
                app.popout_id = sanitized
                status = f"✅ Popout ID saved: {sanitized}"
            app.logger.log(status, level="info")
            return _popout_url(request, app), app.popout_id, status

        def generate_random_popout(value, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.popout_id = secrets.token_urlsafe(16)
            status = f"🎲 New random popout ID saved: {app.popout_id}"
            return _popout_url(request, app), app.popout_id, status

        def test_whisper_connection(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            host = app.settings["whisper_host"]
            if not host:
                return "❌ Please enter a Whisper API host URL"
            headers = {}
            if app.settings.get("whisper_api_key"):
                headers["Authorization"] = f"Bearer {app.settings['whisper_api_key']}"
            try:
                r = requests.get(host, headers=headers, timeout=5)
                if r.status_code == 200:
                    return f"✅ Connected to {host}"
                for ep in ["/health", "/models", "/audio/transcriptions"]:
                    try:
                        r2 = requests.get(
                            host.rstrip("/") + ep, headers=headers, timeout=5
                        )
                        if r2.status_code == 200:
                            return f"✅ Connected to {host.rstrip('/')}{ep}"
                    except Exception:
                        continue
                return "⚠️ Whisper API responds but could not verify endpoint."
            except requests.exceptions.ConnectionError:
                return f"❌ Cannot connect to {host}"
            except Exception as exc:
                return f"❌ Error: {exc}"

        def refresh_sessions_list(request: gr.Request):
            current = get_slug(request)
            try:
                r = requests.get(
                    f"http://{args.host}:{args.port}/active_sessions", timeout=2
                )
                if r.status_code == 200:
                    sessions = r.json().get("sessions", [])
                    choices = [("All sessions", "ALL")] + [
                        (s, s) for s in sessions if s != current
                    ]
                    return (
                        gr.update(choices=choices, value=None),
                        gr.update(
                            value=f"### 🎯 Session: `{current}` | Active: {len(sessions)}"
                        ),
                    )
            except Exception:
                pass
            return gr.update(choices=[]), gr.update()

        def close_session_action(selected, request: gr.Request):
            current = get_slug(request)
            if not selected:
                return "No session selected", gr.update()
            if selected == "ALL":
                with SESSION_LOCK:
                    for h, a in list(SESSION_APPS.items()):
                        if h != current:
                            a.deactivate_session()
                            del SESSION_APPS[h]
                total = len(SESSION_APPS)
                msg = "All other sessions closed"
            else:
                with SESSION_LOCK:
                    a = SESSION_APPS.get(selected)
                    if a:
                        a.deactivate_session()
                        del SESSION_APPS[selected]
                total = len(SESSION_APPS)
                msg = f"Session {selected} closed"
            return msg, gr.update(
                value=f"### 🎯 Session: `{current}` | Active: {total}"
            )

        def start_rec(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            model = (
                app.settings["vosk_model"]
                if app.settings["recognition_engine"] == "vosk"
                else None
            )
            return app.start_recognition(model, app.settings.get("microphone"))

        def stop_rec(request: gr.Request):
            return get_or_create_app(get_slug(request)).stop_recognition()

        def cleanup_user_data(request: gr.Request):
            # Tabs closing / reloading no longer destroys the session — a
            # session now lives until the user explicitly closes it (via the
            # "Manage Sessions" dropdown) or the idle reaper reclaims it after
            # IDLE_SESSION_TIMEOUT_SECONDS of inactivity with nothing running.
            # This is what makes a reload/reconnect re-attach cleanly instead
            # of losing state — which is also what fixed the "sometimes it
            # just refreshes" symptom: it used to tear the whole session down.
            #
            # Gradio fires this on *any* heartbeat drop, not only a real tab
            # close — the tab usually reconnects with the same session_hash
            # a moment later. So the session_hash -> slug mapping is kept
            # (session.py bounds its size); forgetting it here used to leave
            # the tab controlling a ghost session after a network blip.
            slug = get_slug(request)
            app = SESSION_APPS.get(slug)
            if app:
                app.touch()

        def handle_ui_load(js_slug, request: gr.Request):
            # js_slug came from window.location/sessionStorage in the browser
            # (see the JS below) — genuinely scoped to this one tab, unlike a
            # cookie. Register it against this tab's stable session_hash so
            # every *other* handler's plain get_slug(request) call resolves
            # to the same, correct slug for the rest of this tab's life.
            register_slug(request.session_hash, js_slug)
            slug = get_slug(request)
            app = get_or_create_app(slug)

            # Ensure microphone and model are set to valid values
            models = get_available_models()
            mics = get_microphones()
            if models and not app.settings["vosk_model"]:
                app.settings["vosk_model"] = models[0][1]
            if mics and app.settings.get("microphone") not in [m[1] for m in mics]:
                app.settings["microphone"] = mics[0][1]

            s = app.settings
            font_value = s.get("custom_font") or s["font_family"]
            session_html = (
                f'<div id="session-data" data-session="{slug}" '
                f'data-ws-path="/ws/{slug}" '
                f'data-popout="{app.popout_id}">'
                f"<b>Session: {slug} — bookmark this URL to return to it</b></div>"
            )

            return {
                session_info: f"### 🎯 Session: `{slug}` | Active: {len(SESSION_APPS)}",
                popout_url: _popout_url(request, app),
                # Fresh choices, not the list from server start-up — a stale
                # list caused "Value: X is not in the list of choices".
                vosk_model_dropdown: gr.update(choices=models, value=s["vosk_model"]),
                mic_dropdown: gr.update(
                    choices=mics, value=s.get("microphone") if mics else None
                ),
                recognition_engine: s["recognition_engine"],
                audio_mode: s["audio_mode"],
                enable_translation: s["enable_translation"],
                display_interim: s["display_interim"],
                translation_mode: s["translation_mode"],
                source_lang: s["source_language"],
                target_lang: s["target_language"],
                font_selector: font_value,
                recognized_font_size: s["recognized_font_size"],
                translated_font_size: s["translated_font_size"],
                recognized_color: s["recognized_color"],
                translated_color: s["translated_color"],
                background_color: s["background_color"],
                text_alignment: s["text_alignment"],
                vertical_alignment: s["vertical_alignment"],
                translation_position: s["translation_position"],
                whisper_host: s["whisper_host"],
                whisper_api_key: s["whisper_api_key"],
                whisper_model: s["whisper_model"],
                whisper_language: s["whisper_language"],
                whisper_temperature: s["whisper_temperature"],
                whisper_best_of: s["whisper_best_of"],
                whisper_beam_size: s["whisper_beam_size"],
                whisper_patience: s["whisper_patience"],
                whisper_length_penalty: s["whisper_length_penalty"],
                whisper_suppress_tokens: s["whisper_suppress_tokens"],
                whisper_initial_prompt: s["whisper_initial_prompt"],
                whisper_condition_on_previous_text: s[
                    "whisper_condition_on_previous_text"
                ],
                whisper_temperature_increment_on_fallback: s[
                    "whisper_temperature_increment_on_fallback"
                ],
                whisper_no_speech_threshold: s["whisper_no_speech_threshold"],
                whisper_logprob_threshold: s["whisper_logprob_threshold"],
                whisper_compression_ratio_threshold: s[
                    "whisper_compression_ratio_threshold"
                ],
                whisper_endpoint_url: s["whisper_endpoint_url"],
                whisper_response_text_path: s["whisper_response_text_path"],
                whisper_trans_host: s["whisper_translate_host"],
                whisper_trans_api_key: s["whisper_translate_api_key"],
                whisper_trans_model: s["whisper_translate_model"],
                whisper_trans_temperature: s["whisper_translate_temperature"],
                whisper_trans_best_of: s["whisper_translate_best_of"],
                whisper_trans_beam_size: s["whisper_translate_beam_size"],
                whisper_trans_patience: s["whisper_translate_patience"],
                whisper_trans_length_penalty: s["whisper_translate_length_penalty"],
                whisper_trans_suppress_tokens: s["whisper_translate_suppress_tokens"],
                whisper_trans_initial_prompt: s["whisper_translate_initial_prompt"],
                whisper_trans_condition_on_previous_text: s[
                    "whisper_translate_condition_on_previous_text"
                ],
                whisper_trans_temperature_increment_on_fallback: s[
                    "whisper_translate_temperature_increment_on_fallback"
                ],
                whisper_trans_no_speech_threshold: s[
                    "whisper_translate_no_speech_threshold"
                ],
                whisper_trans_logprob_threshold: s[
                    "whisper_translate_logprob_threshold"
                ],
                whisper_trans_compression_ratio_threshold: s[
                    "whisper_translate_compression_ratio_threshold"
                ],
                whisper_trans_endpoint_url: s["whisper_translate_endpoint_url"],
                whisper_trans_response_text_path: s[
                    "whisper_translate_response_text_path"
                ],
                argos_source: s["argos_source_lang"],
                argos_target: s["argos_target_lang"],
                fade_timeout: s["fade_timeout"],
                libretranslate_host: s["libretranslate_host"],
                libretranslate_api_key: s["libretranslate_api_key"],
                ai_host: s["ai_host"],
                ai_api_key: s["ai_api_key"],
                ai_model: s["ai_model"],
                ai_translation_prompt_template: s["ai_translation_prompt_template"],
                ai_endpoint_url: s["ai_endpoint_url"],
                ai_request_body_template: s["ai_request_body_template"],
                ai_response_text_path: s["ai_response_text_path"],
                outline_width: s["outline_width"],
                outline_color: s["outline_color"],
                translated_outline_width: s["translated_outline_width"],
                translated_outline_color: s["translated_outline_color"],
                vad_threshold: _migrate_vad_threshold(s["vad_threshold"]),
                vad_end_silence_ms: s["vad_end_silence_ms"],
                vad_min_speech_ms: s["vad_min_speech_ms"],
                subtitle_mode: s["subtitle_mode"],
                subtitle_cps: s["subtitle_cps"],
                subtitle_max_lines: s["subtitle_max_lines"],
                moonshine_language: s["moonshine_language"],
                moonshine_cache_dir: s["moonshine_cache_dir"],
                browser_session_data: session_html,
                browser_status: "Ready",
                custom_popout_id: app.popout_id,
                session_dropdown: gr.update(choices=[]),
                noise_filter_threshold: s["noise_filter_threshold"],
                audio_watchdog_seconds: s["audio_watchdog_seconds"],
                whisper_low_latency: s["whisper_low_latency"],
                whisper_interim: s["whisper_interim"],
                whisper_max_segment_s: s["whisper_max_segment_s"],
            }

        def reset_to_defaults(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            # Overwrite with defaults, but keep slug, popout_id, etc. – those are not in DEFAULT_SETTINGS
            app.settings.update(VoiceTranslatorApp.DEFAULT_SETTINGS)
            # Also update subtitle manager and VAD with the new values
            app.apply_subtitle_settings()
            app.apply_vad_settings()

            # Re-query what's actually on disk/available right now — same
            # reasoning as handle_ui_load: a stale choices= list from
            # server-startup time is what caused "Value: X is not in the
            # list of choices" after downloading a model and refreshing.
            models = get_available_models()
            mics = get_microphones()
            if models and not app.settings["vosk_model"]:
                app.settings["vosk_model"] = models[0][1]
            if mics and app.settings.get("microphone") not in [m[1] for m in mics]:
                app.settings["microphone"] = mics[0][1]
            persist_settings(app.slug, app.settings)

            # Build the same output dictionary as handle_ui_load
            s = app.settings
            return {
                vosk_model_dropdown: gr.update(choices=models, value=s["vosk_model"]),
                mic_dropdown: gr.update(
                    choices=mics, value=s.get("microphone") if mics else None
                ),
                recognition_engine: s["recognition_engine"],
                audio_mode: s["audio_mode"],
                enable_translation: s["enable_translation"],
                display_interim: s["display_interim"],
                translation_mode: s["translation_mode"],
                source_lang: s["source_language"],
                target_lang: s["target_language"],
                font_selector: s.get("custom_font") or s["font_family"],
                recognized_font_size: s["recognized_font_size"],
                translated_font_size: s["translated_font_size"],
                recognized_color: s["recognized_color"],
                translated_color: s["translated_color"],
                background_color: s["background_color"],
                text_alignment: s["text_alignment"],
                vertical_alignment: s["vertical_alignment"],
                translation_position: s["translation_position"],
                whisper_host: s["whisper_host"],
                whisper_api_key: s["whisper_api_key"],
                whisper_model: s["whisper_model"],
                whisper_language: s["whisper_language"],
                whisper_temperature: s["whisper_temperature"],
                whisper_best_of: s["whisper_best_of"],
                whisper_beam_size: s["whisper_beam_size"],
                whisper_patience: s["whisper_patience"],
                whisper_length_penalty: s["whisper_length_penalty"],
                whisper_suppress_tokens: s["whisper_suppress_tokens"],
                whisper_initial_prompt: s["whisper_initial_prompt"],
                whisper_condition_on_previous_text: s[
                    "whisper_condition_on_previous_text"
                ],
                whisper_temperature_increment_on_fallback: s[
                    "whisper_temperature_increment_on_fallback"
                ],
                whisper_no_speech_threshold: s["whisper_no_speech_threshold"],
                whisper_logprob_threshold: s["whisper_logprob_threshold"],
                whisper_compression_ratio_threshold: s[
                    "whisper_compression_ratio_threshold"
                ],
                whisper_endpoint_url: s["whisper_endpoint_url"],
                whisper_response_text_path: s["whisper_response_text_path"],
                whisper_trans_host: s["whisper_translate_host"],
                whisper_trans_api_key: s["whisper_translate_api_key"],
                whisper_trans_model: s["whisper_translate_model"],
                whisper_trans_temperature: s["whisper_translate_temperature"],
                whisper_trans_best_of: s["whisper_translate_best_of"],
                whisper_trans_beam_size: s["whisper_translate_beam_size"],
                whisper_trans_patience: s["whisper_translate_patience"],
                whisper_trans_length_penalty: s["whisper_translate_length_penalty"],
                whisper_trans_suppress_tokens: s["whisper_translate_suppress_tokens"],
                whisper_trans_initial_prompt: s["whisper_translate_initial_prompt"],
                whisper_trans_condition_on_previous_text: s[
                    "whisper_translate_condition_on_previous_text"
                ],
                whisper_trans_temperature_increment_on_fallback: s[
                    "whisper_translate_temperature_increment_on_fallback"
                ],
                whisper_trans_no_speech_threshold: s[
                    "whisper_translate_no_speech_threshold"
                ],
                whisper_trans_logprob_threshold: s[
                    "whisper_translate_logprob_threshold"
                ],
                whisper_trans_compression_ratio_threshold: s[
                    "whisper_translate_compression_ratio_threshold"
                ],
                whisper_trans_endpoint_url: s["whisper_translate_endpoint_url"],
                whisper_trans_response_text_path: s[
                    "whisper_translate_response_text_path"
                ],
                argos_source: s["argos_source_lang"],
                argos_target: s["argos_target_lang"],
                fade_timeout: s["fade_timeout"],
                libretranslate_host: s["libretranslate_host"],
                libretranslate_api_key: s["libretranslate_api_key"],
                ai_host: s["ai_host"],
                ai_api_key: s["ai_api_key"],
                ai_model: s["ai_model"],
                ai_translation_prompt_template: s["ai_translation_prompt_template"],
                ai_endpoint_url: s["ai_endpoint_url"],
                ai_request_body_template: s["ai_request_body_template"],
                ai_response_text_path: s["ai_response_text_path"],
                outline_width: s["outline_width"],
                outline_color: s["outline_color"],
                translated_outline_width: s["translated_outline_width"],
                translated_outline_color: s["translated_outline_color"],
                vad_threshold: s["vad_threshold"],
                vad_end_silence_ms: s["vad_end_silence_ms"],
                vad_min_speech_ms: s["vad_min_speech_ms"],
                subtitle_mode: s["subtitle_mode"],
                subtitle_cps: s["subtitle_cps"],
                subtitle_max_lines: s["subtitle_max_lines"],
                moonshine_language: s["moonshine_language"],
                moonshine_cache_dir: s["moonshine_cache_dir"],
                noise_filter_threshold: s["noise_filter_threshold"],
                audio_watchdog_seconds: s["audio_watchdog_seconds"],
                whisper_low_latency: s["whisper_low_latency"],
                whisper_interim: s["whisper_interim"],
                whisper_max_segment_s: s["whisper_max_segment_s"],
            }

        # ── Wire events ───────────────────────────────────────────────────────

        recognition_engine.change(
            update_recognition_engine,
            [recognition_engine],
            [vosk_settings, whisper_settings, moonshine_settings],
        )
        audio_mode.change(
            update_audio_mode,
            [audio_mode],
            [hardware_group, browser_group, browser_source_hint],
        )
        translation_mode.change(
            update_translation_mode,
            [translation_mode],
            [
                ai_settings,
                libretranslate_settings,
                whisper_translate_settings,
                argos_settings,
                source_lang,
                target_lang,
            ],
        )
        enable_translation.change(
            update_translation_toggle, [enable_translation], [translation_group]
        )
        display_interim.change(_set("display_interim"), [display_interim])

        # Language / translation
        source_lang.change(_set("source_language"), [source_lang])
        target_lang.change(_set("target_language"), [target_lang])
        argos_source.change(_set("argos_source_lang"), [argos_source])
        argos_target.change(_set("argos_target_lang"), [argos_target])
        ai_host.change(_set("ai_host"), [ai_host])
        ai_api_key.change(_set("ai_api_key"), [ai_api_key])
        ai_model.change(_set("ai_model"), [ai_model])
        ai_translation_prompt_template.change(
            _set("ai_translation_prompt_template"), [ai_translation_prompt_template]
        )
        ai_endpoint_url.change(_set("ai_endpoint_url"), [ai_endpoint_url])
        ai_request_body_template.change(
            _set("ai_request_body_template"), [ai_request_body_template]
        )
        ai_response_text_path.change(
            _set("ai_response_text_path"), [ai_response_text_path]
        )

        def reset_ai_advanced(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            for key in (
                "ai_translation_prompt_template",
                "ai_endpoint_url",
                "ai_request_body_template",
                "ai_response_text_path",
            ):
                app.settings[key] = ""
            persist_settings(app.slug, app.settings)
            return "", "", "", ""

        ai_advanced_reset_btn.click(
            reset_ai_advanced,
            outputs=[
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
            ],
        )
        libretranslate_host.change(_set("libretranslate_host"), [libretranslate_host])
        libretranslate_api_key.change(
            _set("libretranslate_api_key"), [libretranslate_api_key]
        )

        # Whisper recognition
        whisper_host.change(_set("whisper_host"), [whisper_host])
        whisper_api_key.change(_set("whisper_api_key"), [whisper_api_key])
        whisper_model.change(_set("whisper_model"), [whisper_model])
        whisper_language.change(_set("whisper_language"), [whisper_language])
        whisper_temperature.change(_set("whisper_temperature"), [whisper_temperature])
        whisper_best_of.change(_set("whisper_best_of"), [whisper_best_of])
        whisper_beam_size.change(_set("whisper_beam_size"), [whisper_beam_size])
        whisper_patience.change(_set("whisper_patience"), [whisper_patience])
        whisper_length_penalty.change(
            _set("whisper_length_penalty"), [whisper_length_penalty]
        )
        whisper_suppress_tokens.change(
            _set("whisper_suppress_tokens"), [whisper_suppress_tokens]
        )
        whisper_initial_prompt.change(
            _set("whisper_initial_prompt"), [whisper_initial_prompt]
        )
        whisper_condition_on_previous_text.change(
            _set("whisper_condition_on_previous_text"),
            [whisper_condition_on_previous_text],
        )
        whisper_temperature_increment_on_fallback.change(
            _set("whisper_temperature_increment_on_fallback"),
            [whisper_temperature_increment_on_fallback],
        )
        whisper_no_speech_threshold.change(
            _set("whisper_no_speech_threshold"), [whisper_no_speech_threshold]
        )
        whisper_logprob_threshold.change(
            _set("whisper_logprob_threshold"), [whisper_logprob_threshold]
        )
        whisper_compression_ratio_threshold.change(
            _set("whisper_compression_ratio_threshold"),
            [whisper_compression_ratio_threshold],
        )
        whisper_endpoint_url.change(
            _set("whisper_endpoint_url"), [whisper_endpoint_url]
        )
        whisper_response_text_path.change(
            _set("whisper_response_text_path"), [whisper_response_text_path]
        )

        # Whisper translate
        whisper_trans_host.change(_set("whisper_translate_host"), [whisper_trans_host])
        whisper_trans_api_key.change(
            _set("whisper_translate_api_key"), [whisper_trans_api_key]
        )
        whisper_trans_model.change(
            _set("whisper_translate_model"), [whisper_trans_model]
        )
        whisper_trans_temperature.change(
            _set("whisper_translate_temperature"), [whisper_trans_temperature]
        )
        whisper_trans_best_of.change(
            _set("whisper_translate_best_of"), [whisper_trans_best_of]
        )
        whisper_trans_beam_size.change(
            _set("whisper_translate_beam_size"), [whisper_trans_beam_size]
        )
        whisper_trans_patience.change(
            _set("whisper_translate_patience"), [whisper_trans_patience]
        )
        whisper_trans_length_penalty.change(
            _set("whisper_translate_length_penalty"), [whisper_trans_length_penalty]
        )
        whisper_trans_suppress_tokens.change(
            _set("whisper_translate_suppress_tokens"), [whisper_trans_suppress_tokens]
        )
        whisper_trans_initial_prompt.change(
            _set("whisper_translate_initial_prompt"), [whisper_trans_initial_prompt]
        )
        whisper_trans_condition_on_previous_text.change(
            _set("whisper_translate_condition_on_previous_text"),
            [whisper_trans_condition_on_previous_text],
        )
        whisper_trans_temperature_increment_on_fallback.change(
            _set("whisper_translate_temperature_increment_on_fallback"),
            [whisper_trans_temperature_increment_on_fallback],
        )
        whisper_trans_no_speech_threshold.change(
            _set("whisper_translate_no_speech_threshold"),
            [whisper_trans_no_speech_threshold],
        )
        whisper_trans_logprob_threshold.change(
            _set("whisper_translate_logprob_threshold"),
            [whisper_trans_logprob_threshold],
        )
        whisper_trans_compression_ratio_threshold.change(
            _set("whisper_translate_compression_ratio_threshold"),
            [whisper_trans_compression_ratio_threshold],
        )
        whisper_trans_endpoint_url.change(
            _set("whisper_translate_endpoint_url"), [whisper_trans_endpoint_url]
        )
        whisper_trans_response_text_path.change(
            _set("whisper_translate_response_text_path"),
            [whisper_trans_response_text_path],
        )

        # Display style
        font_selector.change(update_font_selector, [font_selector])
        recognized_font_size.change(
            _set("recognized_font_size"), [recognized_font_size]
        )
        translated_font_size.change(
            _set("translated_font_size"), [translated_font_size]
        )
        recognized_color.change(_set("recognized_color"), [recognized_color])
        translated_color.change(_set("translated_color"), [translated_color])
        background_color.change(_set("background_color"), [background_color])
        text_alignment.change(_set("text_alignment"), [text_alignment])
        vertical_alignment.change(_set("vertical_alignment"), [vertical_alignment])
        translation_position.change(
            _set("translation_position"), [translation_position]
        )
        fade_timeout.change(_set("fade_timeout"), [fade_timeout])
        outline_width.change(_set("outline_width"), [outline_width])
        outline_color.change(_set("outline_color"), [outline_color])
        translated_outline_width.change(
            _set("translated_outline_width"), [translated_outline_width]
        )
        translated_outline_color.change(
            _set("translated_outline_color"), [translated_outline_color]
        )

        # Fade — also syncs subtitle manager
        fade_timeout.change(_set_fade, [fade_timeout])

        # Audio / mic
        mic_dropdown.change(_set("microphone"), [mic_dropdown])
        vosk_model_dropdown.change(_set("vosk_model"), [vosk_model_dropdown])

        # VAD — dynamic, no restart needed
        vad_threshold.change(_set_vad("vad_threshold"), [vad_threshold])
        vad_end_silence_ms.change(_set_vad("vad_end_silence_ms"), [vad_end_silence_ms])
        vad_min_speech_ms.change(_set_vad("vad_min_speech_ms"), [vad_min_speech_ms])
        noise_filter_threshold.change(
            _set_vad("noise_filter_threshold"), [noise_filter_threshold]
        )
        audio_watchdog_seconds.change(
            _set("audio_watchdog_seconds"), [audio_watchdog_seconds]
        )
        whisper_low_latency.change(_set("whisper_low_latency"), [whisper_low_latency])
        whisper_interim.change(_set("whisper_interim"), [whisper_interim])
        whisper_max_segment_s.change(
            _set_vad("whisper_max_segment_s"), [whisper_max_segment_s]
        )

        # Mic test
        test_mic_btn.click(
            start_mic_test,
            outputs=[mic_test_status, stop_test_mic_btn, test_mic_btn, mic_test_status],
            js="startHwLevelPolling",
        )
        stop_test_mic_btn.click(
            stop_mic_test,
            outputs=[mic_test_status, stop_test_mic_btn, test_mic_btn, mic_test_status],
            js="stopHwLevelPolling",
        )

        browser_test_mic_btn.click(
            start_browser_mic_test,
            outputs=[browser_status, browser_stop_test_mic_btn, browser_test_mic_btn],
            js="startBrowserMicTest",
        )
        browser_stop_test_mic_btn.click(
            stop_browser_mic_test,
            outputs=[browser_status, browser_stop_test_mic_btn, browser_test_mic_btn],
            js="stopBrowserMicTest",
        )

        # Subtitle timing
        subtitle_mode.change(
            update_subtitle_mode, [subtitle_mode], [subtitle_buffered_group]
        )
        subtitle_cps.change(_set_subtitle("subtitle_cps"), [subtitle_cps])
        subtitle_max_lines.change(
            _set_subtitle("subtitle_max_lines"), [subtitle_max_lines]
        )

        # Moonshine
        moonshine_language.change(_set("moonshine_language"), [moonshine_language])
        moonshine_cache_dir.change(_set("moonshine_cache_dir"), [moonshine_cache_dir])

        # Popout
        custom_popout_id.submit(
            update_custom_popout_id,
            [custom_popout_id],
            [popout_url, custom_popout_id, status_text],
        )
        # Also save when the field loses focus — pressing Enter used to be the
        # only way, and the id was never written to disk either way.
        custom_popout_id.blur(
            update_custom_popout_id,
            [custom_popout_id],
            [popout_url, custom_popout_id, status_text],
        )
        random_btn.click(
            generate_random_popout,
            [random_btn],
            [popout_url, custom_popout_id, status_text],
        )

        # Session management
        _OPEN_SESSION_JS = """
        (name) => {
            name = (name || "").trim();
            if (!name) return;
            const clean = name.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
            if (!clean) {
                alert("Session name can only contain letters, numbers, - and _");
                return;
            }
            window.location.href = "/?session=" + encodeURIComponent(clean);
        }
        """
        open_session_btn.click(fn=None, inputs=[new_session_name], js=_OPEN_SESSION_JS)
        new_session_name.submit(
            fn=None, inputs=[new_session_name], js=_OPEN_SESSION_JS
        )
        random_session_btn.click(
            fn=None,
            js="""
            () => {
                const rand = Math.random().toString(36).slice(2, 10);
                window.location.href = "/?session=" + rand;
            }
            """,
        )
        refresh_sessions_btn.click(
            refresh_sessions_list, outputs=[session_dropdown, session_info]
        )
        # Selecting a session in the dropdown used to close it immediately on
        # .change() — surprising ("I cannot close the session" reports were
        # actually the opposite: it was closing on mere selection, before
        # anyone meant to confirm that). Now selection just selects; this
        # button is the actual close action.
        close_session_btn.click(
            close_session_action, [session_dropdown], [status_text, session_info]
        ).then(refresh_sessions_list, outputs=[session_dropdown, session_info])

        # Refresh buttons
        refresh_fonts_btn.click(
            lambda: gr.update(choices=SYSTEM_FONTS + get_available_fonts()),
            outputs=[font_selector],
        )
        refresh_models_btn.click(
            lambda: gr.update(choices=get_available_models()),
            outputs=[vosk_model_dropdown],
        )
        refresh_mic_btn.click(
            lambda: gr.update(choices=get_microphones()), outputs=[mic_dropdown]
        )

        # Whisper test
        test_whisper_btn.click(test_whisper_connection, outputs=[status_text])

        # Start — also starts HW level polling; stops mic test if active
        # (The browser stream status box is driven by the page's JS, which
        # knows whether the mic actually opened / the socket is connected.)
        start_btn.click(
            fn=start_rec, outputs=[status_text], js="startBrowserStreaming"
        ).then(
            fn=None, inputs=[status_text], js="(s) => { window.vtAfterStart(s); }"
        ).then(
            fn=lambda: gr.update(visible=False), outputs=[stop_test_mic_btn]
        ).then(fn=None, js="startHwLevelPolling")

        # Stop — stop HW polling too
        stop_btn.click(
            fn=stop_rec, outputs=[status_text], js="stopBrowserStreaming"
        ).then(fn=None, js="stopHwLevelPolling")

        reset_defaults_btn.click(
            reset_to_defaults,
            outputs=[
                vosk_model_dropdown,
                mic_dropdown,
                recognition_engine,
                audio_mode,
                enable_translation,
                display_interim,
                translation_mode,
                source_lang,
                target_lang,
                font_selector,
                recognized_font_size,
                translated_font_size,
                recognized_color,
                translated_color,
                background_color,
                text_alignment,
                vertical_alignment,
                translation_position,
                whisper_host,
                whisper_api_key,
                whisper_model,
                whisper_language,
                whisper_temperature,
                whisper_best_of,
                whisper_beam_size,
                whisper_patience,
                whisper_length_penalty,
                whisper_suppress_tokens,
                whisper_initial_prompt,
                whisper_condition_on_previous_text,
                whisper_temperature_increment_on_fallback,
                whisper_no_speech_threshold,
                whisper_logprob_threshold,
                whisper_compression_ratio_threshold,
                whisper_endpoint_url,
                whisper_response_text_path,
                whisper_trans_host,
                whisper_trans_api_key,
                whisper_trans_model,
                whisper_trans_temperature,
                whisper_trans_best_of,
                whisper_trans_beam_size,
                whisper_trans_patience,
                whisper_trans_length_penalty,
                whisper_trans_suppress_tokens,
                whisper_trans_initial_prompt,
                whisper_trans_condition_on_previous_text,
                whisper_trans_temperature_increment_on_fallback,
                whisper_trans_no_speech_threshold,
                whisper_trans_logprob_threshold,
                whisper_trans_compression_ratio_threshold,
                whisper_trans_endpoint_url,
                whisper_trans_response_text_path,
                argos_source,
                argos_target,
                fade_timeout,
                libretranslate_host,
                libretranslate_api_key,
                ai_host,
                ai_api_key,
                ai_model,
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
                outline_width,
                outline_color,
                translated_outline_width,
                translated_outline_color,
                vad_threshold,
                vad_end_silence_ms,
                vad_min_speech_ms,
                subtitle_mode,
                subtitle_cps,
                subtitle_max_lines,
                moonshine_language,
                moonshine_cache_dir,
                noise_filter_threshold,
                audio_watchdog_seconds,
                whisper_low_latency,
                whisper_interim,
                whisper_max_segment_s,
            ],
        )

        # ── Display / logs / session-state polling ────────────────────────────
        # Done by the page's own JS (startAllPolling) with plain fetch() calls
        # to /display_data and /logs_data. This used to be a gr.Timer(0.05):
        # 20 Gradio queue events per second per open tab, all through the
        # same connection as every button click. When anything slowed that
        # connection (a backgrounded tab, a proxy, a busy server) the backlog
        # made the page stall and reconnect — the "refresh" that cut off
        # the mic stream.

        interface.unload(cleanup_user_data)

        # Resolves this tab's slug purely client-side (query string, or a
        # value persisted in sessionStorage — which, unlike cookies/
        # localStorage, is genuinely scoped to one tab) before handle_ui_load
        # runs. See session.py's module docstring for why this replaced the
        # old cookie-based approach.
        _RESOLVE_SLUG_JS = """
        () => {
            const params = new URLSearchParams(window.location.search);
            let slug = params.get('session');
            if (!slug) {
                slug = sessionStorage.getItem('vt_slug');
            }
            if (!slug) {
                slug = 'main';
            }
            slug = slug.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 64) || 'main';
            sessionStorage.setItem('vt_slug', slug);
            // Keep the session name in the address bar. Besides making the
            // URL bookmarkable, the browser sends it (as the Referer) with
            // every Gradio request, which lets the server recover this tab's
            // session if it ever loses track of it (e.g. after a restart).
            if (params.get('session') !== slug && window.location.pathname === '/') {
                params.set('session', slug);
                history.replaceState(null, '', '/?' + params.toString() + window.location.hash);
            }
            return [slug];
        }
        """

        interface.load(
            fn=None,
            inputs=None,
            outputs=[slug_state],
            js=_RESOLVE_SLUG_JS,
        ).then(
            handle_ui_load,
            inputs=[slug_state],
            outputs=[
                session_info,
                popout_url,
                vosk_model_dropdown,
                mic_dropdown,
                recognition_engine,
                audio_mode,
                enable_translation,
                display_interim,
                translation_mode,
                source_lang,
                target_lang,
                font_selector,
                recognized_font_size,
                translated_font_size,
                recognized_color,
                translated_color,
                background_color,
                text_alignment,
                vertical_alignment,
                translation_position,
                whisper_host,
                whisper_api_key,
                whisper_model,
                whisper_language,
                whisper_temperature,
                whisper_best_of,
                whisper_beam_size,
                whisper_patience,
                whisper_length_penalty,
                whisper_suppress_tokens,
                whisper_initial_prompt,
                whisper_condition_on_previous_text,
                whisper_temperature_increment_on_fallback,
                whisper_no_speech_threshold,
                whisper_logprob_threshold,
                whisper_compression_ratio_threshold,
                whisper_endpoint_url,
                whisper_response_text_path,
                whisper_trans_host,
                whisper_trans_api_key,
                whisper_trans_model,
                whisper_trans_temperature,
                whisper_trans_best_of,
                whisper_trans_beam_size,
                whisper_trans_patience,
                whisper_trans_length_penalty,
                whisper_trans_suppress_tokens,
                whisper_trans_initial_prompt,
                whisper_trans_condition_on_previous_text,
                whisper_trans_temperature_increment_on_fallback,
                whisper_trans_no_speech_threshold,
                whisper_trans_logprob_threshold,
                whisper_trans_compression_ratio_threshold,
                whisper_trans_endpoint_url,
                whisper_trans_response_text_path,
                argos_source,
                argos_target,
                fade_timeout,
                libretranslate_host,
                libretranslate_api_key,
                ai_host,
                ai_api_key,
                ai_model,
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
                outline_width,
                outline_color,
                translated_outline_width,
                translated_outline_color,
                vad_threshold,
                vad_end_silence_ms,
                vad_min_speech_ms,
                subtitle_mode,
                subtitle_cps,
                subtitle_max_lines,
                moonshine_language,
                moonshine_cache_dir,
                browser_session_data,
                browser_status,
                custom_popout_id,
                session_dropdown,
                noise_filter_threshold,
                audio_watchdog_seconds,
                whisper_low_latency,
                whisper_interim,
                whisper_max_segment_s,
            ],
        )

    return interface


# ── Logging config ────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "level": "WARNING",
            "handlers": ["access"],
            "propagate": False,
        },
    },
}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Resolves the persistent session slug (see top of file) and pins it to
    # the browser via a cookie. Must be added before mount_gradio_app below.
    fastapi_app.add_middleware(SessionSlugMiddleware)

    @fastapi_app.get("/mic_level/{slug}")
    async def get_mic_level(slug: str):
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            return JSONResponse({"rms": app.monitor_level})
        return JSONResponse({"rms": 0.0})

    @fastapi_app.get("/display_data/{slug}")
    async def get_display_data(slug: str):
        """
        Polled directly by JS every 50 ms — completely bypasses Gradio's SSE queue.
        This is what makes the display update without ever triggering a page reload.
        """
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            html, rec, trans = app.get_current_display()
            return JSONResponse(
                {
                    "html": html,
                    "recognized": rec,
                    "translated": trans,
                    # Session state, so every open tab notices an automatic
                    # stop (audio source lost) and can re-attach after a reload.
                    "running": app.is_running,
                    "audio_mode": app.settings.get("audio_mode"),
                    "audio_client": app.has_audio_client,
                    "stop_reason": app.stop_reason,
                    "state_seq": app.state_seq,
                }
            )
        return JSONResponse(
            {"html": "", "recognized": "", "translated": "", "running": False}
        )

    @fastapi_app.get("/logs_data/{slug}")
    async def get_logs_data(slug: str):
        """Polled by JS every 2 s — serves log text without using Gradio SSE."""
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            return JSONResponse({"logs": app.update_logs()})
        return JSONResponse({"logs": ""})

    @fastapi_app.get("/active_sessions")
    async def get_active_sessions():
        with SESSION_LOCK:
            return JSONResponse({"sessions": list(SESSION_APPS.keys())})

    @fastapi_app.post("/deactivate/{slug}")
    async def deactivate_session(slug: str):
        """
        Explicit, manual full teardown of a session (stops recognition, frees
        models). Not called automatically on tab close/reload anymore — use
        the "Manage Sessions" panel in the UI, or call this directly, when you
        actually want to free a session's resources.
        """
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
            if app:
                app.deactivate_session()
                del SESSION_APPS[slug]
                return JSONResponse({"status": "deactivated"})
        return JSONResponse({"status": "not found"}, status_code=404)

    @fastapi_app.post("/session_stop/{slug}")
    async def session_stop(slug: str, reason: str = ""):
        """Called by the page when its audio source ends and the WebSocket is down."""
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app and app.is_running:
            await run_in_threadpool(
                app.stop_recognition, reason=_source_ended_reason(reason)
            )
        return JSONResponse({"status": "ok"})

    @fastapi_app.websocket("/ws/{slug}")
    async def websocket_endpoint(websocket: WebSocket, slug: str):
        await websocket.accept()
        app = get_or_create_app(slug)
        conn_id, previous = app.attach_audio_client(
            websocket, asyncio.get_running_loop()
        )
        if previous is not None:
            # Another tab was streaming into this session — tell it to stop
            # (and not reconnect), so the two don't fight over the session.
            try:
                await previous[1].send_text(json.dumps({"type": "replaced"}))
                await previous[1].close()
            except Exception:
                pass
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    app.feed_browser_audio(data, conn_id)
                    continue
                text = msg.get("text")
                if text:
                    try:
                        event = json.loads(text)
                    except ValueError:
                        continue
                    if event.get("type") == "source_ended" and app.is_running:
                        # The browser saw its mic/tab-share end (unplugged,
                        # permission revoked, "Stop sharing" clicked).
                        await run_in_threadpool(
                            app.stop_recognition,
                            reason=_source_ended_reason(event.get("reason", "")),
                        )
        except WebSocketDisconnect:
            # A dropped connection alone doesn't stop the session: the page
            # reconnects by itself after a network blip. If the source is
            # really gone, the audio watchdog stops the session once no
            # audio has arrived for `audio_watchdog_seconds`.
            pass
        except Exception as exc:
            print(f"WebSocket error [{slug}]: {exc}")
        finally:
            app.detach_audio_client(conn_id)

    @fastapi_app.get("/popout/{popout_id}")
    async def get_popout(popout_id: str):
        app = await run_in_threadpool(find_app_by_popout_id, popout_id)
        if app:
            return HTMLResponse(content=app.generate_popout_html())
        return HTMLResponse("<h1>Not found</h1>", 404)

    @fastapi_app.get("/popout_data/{popout_id}")
    async def get_popout_data(popout_id: str):
        app = await run_in_threadpool(find_app_by_popout_id, popout_id)
        if app:
            _, rec, trans = app.get_current_display()
            return JSONResponse(
                {
                    "recognized": rec,
                    "translated": trans if app.settings["enable_translation"] else "",
                    "style_key": app.popout_style_key(),
                }
            )
        return JSONResponse({"error": "Not found"}, 404)

    @fastapi_app.get("/fonts/{filename}")
    async def get_font(filename: str):
        font_path = Path("fonts") / filename
        if not font_path.exists():
            return JSONResponse({"error": "Font not found"}, 404)
        return FileResponse(path=font_path, media_type="font/ttf")

    interface = create_ui(args)
    # Disable Gradio's built-in queue size limit and concurrency cap so it never
    # drops a client connection (which manifests as a page reload).
    interface.queue(
        max_size=None,  # never refuse/drop connections
        default_concurrency_limit=None,  # no cap on parallel event handlers
    )
    fastapi_app = gr.mount_gradio_app(fastapi_app, interface, path="/", head=js)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🎤 Voice Translator - Multi-Session                 ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                                  ║
║  Named session: http://{args.host}:{args.port}/?session=<name>        ║
║                                                              ║
║  ✅ Per-session settings saved under {str(SETTINGS_DIR):<24}║
║  🎚️  VAD default: 50 ms silence timeout (all engines)      ║
║  ⏹️  Stop unloads models, settings preserved               ║
║  🌙 Moonshine: auto-download from HuggingFace               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        log_config=LOG_CONFIG,
        # NOTE: this was previously `0`, which tells uvicorn to close the
        # keep-alive connection almost immediately after each response
        # instead of the "1 hour" the comment claimed. That was very likely
        # the main cause of "sessions randomly refresh" — Gradio's own SSE
        # queue connection (and any long HTTP polling) got dropped every few
        # seconds, forcing the frontend to reconnect. 3600s = 1 hour, as the
        # comment always intended.
        timeout_keep_alive=3600,  # never drop an idle SSE/WS connection
        ws_ping_interval=30,  # keep WebSocket alive with pings every 30 s
        ws_ping_timeout=120,  # give 2 minutes to respond before dropping
    )
