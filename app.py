import argparse
import io
import json
import os
import queue
import re
import secrets
import threading
import time
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from vosk import KaldiRecognizer, Model

from logger import Logger
from translators import TranslationService

# Load environment variables
os.environ["ARGOS_PACKAGES_DIR"] = os.getcwd() + "/argos_models"

# Try to import langdetect for auto language detection
try:
    from langdetect import LangDetectException, detect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARNING] langdetect not installed.")
    print("Install with: pip install langdetect")

# Try to import argostranslate
try:
    import argostranslate.package
    import argostranslate.translate

    ARGOS_AVAILABLE = True
except ImportError:
    ARGOS_AVAILABLE = False
    print("[WARNING] argostranslate not installed. Offline translation disabled.")
    print("Install with: pip install argostranslate")

# Global storage - ONE app instance per session_hash
SESSION_APPS = {}
SESSION_LOCK = threading.Lock()


def get_available_fonts():
    """Scan the fonts/ directory for supported font files."""
    fonts_dir = Path("fonts")
    if not fonts_dir.exists():
        fonts_dir.mkdir(exist_ok=True)
        return []
    fonts = []
    for ext in (".ttf", ".otf", ".woff", ".woff2"):
        for file in fonts_dir.glob(f"*{ext}"):
            fonts.append(
                (f"[Custom] {file.stem}", file.name)
            )  # display name, value = filename
    return fonts


# System fonts list (you can expand this)
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

js = """
<script>
// State variables
window.__audioStreamActive = false;
window.__audioProcessor = null;
window.__audioSource = null;
window.__mediaStream = null;
window.__audioContext = null;
window.__websocket = null;

// Helper to get current gain value from slider
function getGain() {
    var slider = document.getElementById('browser-gain-slider');
    return slider ? parseFloat(slider.value) || 1.0 : 1.0;
}

// Helper to update status textbox
function setStatus(text) {
    var status = document.getElementById('browser-status');
    if (status) status.value = text;
}

// Start streaming function (called by Start button)
window.startBrowserStreaming = function() {
    // Only run if browser mode is selected
    var browserRadio = document.querySelector('input[value="browser"]');
    if (!browserRadio || !browserRadio.checked) return;

    if (window.__audioStreamActive) return;
    window.__audioStreamActive = true;

    var sessionDiv = document.getElementById('session-data');
    if (!sessionDiv) {
        console.error('Session data element not found');
        setStatus('Error: session data missing');
        window.__audioStreamActive = false;
        return;
    }
    var wsUrl = sessionDiv.dataset.wsUrl;
    if (!wsUrl) {
        console.error('WebSocket URL not found');
        setStatus('Error: WebSocket URL missing');
        window.__audioStreamActive = false;
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            window.__mediaStream = stream;
            var AudioContext = window.AudioContext || window.webkitAudioContext;
            window.__audioContext = new AudioContext();
            var inputSampleRate = window.__audioContext.sampleRate;

            window.__audioSource = window.__audioContext.createMediaStreamSource(stream);
            window.__audioProcessor = window.__audioContext.createScriptProcessor(4096, 1, 1);

            window.__audioSource.connect(window.__audioProcessor);
            window.__audioProcessor.connect(window.__audioContext.destination);

            window.__websocket = new WebSocket(wsUrl);
            window.__websocket.binaryType = 'arraybuffer';

            window.__websocket.onopen = function() {
                console.log('WebSocket connected');
                setStatus('Streaming...');
            };

            window.__websocket.onerror = function(err) {
                console.error('WebSocket error', err);
                setStatus('WebSocket error');
                window.stopBrowserStreaming();
            };

            window.__websocket.onclose = function() {
                console.log('WebSocket closed');
                setStatus('Stopped');
                window.__audioStreamActive = false;
            };

            window.__audioProcessor.onaudioprocess = function(event) {
                if (window.__websocket.readyState !== WebSocket.OPEN) return;

                var inputData = event.inputBuffer.getChannelData(0);
                var gain = getGain();

                if (gain !== 1.0) {
                    for (var i = 0; i < inputData.length; i++) {
                        inputData[i] *= gain;
                    }
                }

                var outputData;
                if (inputSampleRate !== 16000) {
                    var ratio = 16000 / inputSampleRate;
                    var outLen = Math.floor(inputData.length * ratio);
                    outputData = new Float32Array(outLen);
                    for (var i = 0; i < outLen; i++) {
                        var pos = i / ratio;
                        var i1 = Math.floor(pos);
                        var i2 = Math.min(i1 + 1, inputData.length - 1);
                        var frac = pos - i1;
                        outputData[i] = inputData[i1] * (1 - frac) + inputData[i2] * frac;
                    }
                } else {
                    outputData = inputData;
                }

                var int16 = new Int16Array(outputData.length);
                for (var i = 0; i < outputData.length; i++) {
                    var s = Math.max(-1, Math.min(1, outputData[i]));
                    s = s < 0 ? s * 32768 : s * 32767;
                    int16[i] = Math.round(s);
                }

                window.__websocket.send(int16.buffer);
            };
        })
        .catch(function(err) {
            console.error('Microphone error:', err);
            setStatus('Microphone error: ' + err.message);
            window.__audioStreamActive = false;
        });
};

// Stop streaming function (called by Stop button)
window.stopBrowserStreaming = function() {
    if (!window.__audioStreamActive) return;
    window.__audioStreamActive = false;

    if (window.__audioProcessor) {
        window.__audioProcessor.disconnect();
        window.__audioProcessor = null;
    }
    if (window.__audioSource) {
        window.__audioSource.disconnect();
        window.__audioSource = null;
    }
    if (window.__mediaStream) {
        window.__mediaStream.getTracks().forEach(function(track) { track.stop(); });
        window.__mediaStream = null;
    }
    if (window.__audioContext) {
        window.__audioContext.close();
        window.__audioContext = null;
    }
    if (window.__websocket) {
        window.__websocket.close();
        window.__websocket = null;
    }
    setStatus('Stopped');
};
</script>
"""


def dots_or_stars(input_str: str, second_arg=None) -> bool:
    if input_str == "." or re.match(r"<\|.*|>", input_str):
        return True
    return False


class ArgosTranslator:
    """Offline translation using Argos Translate"""

    def __init__(self, logger=None):
        self.logger = logger
        self.models_dir = Path("argos_models")
        self.models_dir.mkdir(exist_ok=True)
        if ARGOS_AVAILABLE:
            argostranslate.package.settings.package_data_dir = str(self.models_dir)
            try:
                argostranslate.package.update_package_index()
                argostranslate.package.get_available_packages()
                print("✓ Package index updated")
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        f"Argos package index update failed: {str(e)}", level="warning"
                    )

    def translate(self, text, source_lang, target_lang):
        """Translate text using Argos"""
        if not ARGOS_AVAILABLE:
            return f"[Argos not available: {text}]"
        try:
            installed = argostranslate.translate.get_installed_languages()
            source = target = None

            for lang in installed:
                if lang.code == source_lang or lang.code.startswith(source_lang):
                    source = lang
                if lang.code == target_lang or lang.code.startswith(target_lang):
                    target = lang

            if not source or not target:
                if self.logger:
                    self.logger.log(
                        f"Argos: Language pair {source_lang}->{target_lang} not installed",
                        level="warning",
                    )
                return f"[Model not installed: {text}]"

            # Get translation
            translation = source.get_translation(target)
            if not translation:
                if self.logger:
                    self.logger.log(
                        f"Argos: No translation available for {source_lang}->{target_lang}",
                        level="warning",
                    )
                return f"[No translation available: {text}]"

            result = translation.translate(text)
            if self.logger:
                self.logger.log(f"Argos translated: {text} -> {result}", level="info")

            return result

        except Exception as e:
            if self.logger:
                self.logger.log(f"Argos translation error: {str(e)}", level="error")
            return text

    def get_available_languages(self):
        """Get list of available language pairs"""
        if not ARGOS_AVAILABLE:
            return []

        try:
            installed = argostranslate.translate.get_installed_languages()
            pairs = []
            for source in installed:
                for target in source.translations_from:
                    pairs.append(
                        (source.code, target.code, f"{source.name} -> {target.name}")
                    )
            return pairs
        except Exception as e:
            if self.logger:
                self.logger.log(
                    f"Error getting Argos languages: {str(e)}", level="error"
                )
            return []


class WhisperRecognizer:
    """Whisper API-based speech recognizer"""

    def __init__(self, host, api_key=None, model="whisper-large-v3", logger=None):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def transcribe(self, audio_bytes, sample_rate=16000, language=None):
        try:
            if isinstance(audio_bytes, bytes):
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            else:
                audio_array = audio_bytes

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)

            buffer.seek(0)

            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = {"model": self.model, "response_format": "json", "temperature": 0.0}
            if language:
                data["language"] = language

            response = self.session.post(
                f"{self.host}/audio/transcriptions", files=files, data=data, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict):
                    return result.get("text", "")
                else:
                    return str(result)
            else:
                if self.logger:
                    self.logger.log(
                        f"Whisper API error: {response.status_code} - {response.text}",
                        level="error",
                    )
                return ""

        except Exception as e:
            if self.logger:
                self.logger.log(f"Whisper transcription error: {str(e)}", level="error")
            return ""

    def translate(self, audio_bytes, sample_rate=16000):
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            buffer.seek(0)

            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = {"model": self.model, "response_format": "json"}

            response = self.session.post(
                f"{self.host}/audio/translations", files=files, data=data, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict):
                    return result.get("text", "")
                return str(result)
            else:
                if self.logger:
                    self.logger.log(
                        f"Whisper translation API error: {response.status_code}",
                        level="error",
                    )
                return ""
        except Exception as e:
            if self.logger:
                self.logger.log(f"Whisper translation error: {str(e)}", level="error")
            return ""


class VoiceTranslatorApp:
    def __init__(self, session_hash):
        self.display_running = True
        self.session_hash = session_hash
        self.popout_id = secrets.token_urlsafe(16)
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.recognizer = None
        self.model = None
        self.logger = Logger(session_id=session_hash)
        self.stream = None
        self.whisper_recognizer = None
        self.argos_translator = None

        self.audio_buffer = bytearray()
        self.buffer_duration = 0
        self.target_buffer_duration = 3.0
        self.last_audio_chunk = None

        self.vosk_audio_buffer = bytearray()  # accumulates audio for Vosk mode

        self.current_recognized = "Waiting for speech..."
        self.current_translated = ""
        self.last_update_time = time.time()

        self.current_language = "en"
        self.available_vosk_models = {}
        self.translation_service = None

        self.settings = {
            "model_path": "",
            "microphone": 12,
            "vosk_model": "",
            "audio_mode": "hardware",
            "enable_translation": False,
            "local_language_translation": False,
            "display_interim": True,
            "translation_mode": "argos",
            "source_language": "en-US",
            "target_language": "es",
            "font_family": "Arial",
            "recognized_font_size": 48,
            "translated_font_size": 32,
            "recognized_color": "#FFFFFF",
            "translated_color": "#CCCCCC",
            "background_color": "#000000",
            "text_alignment": "center",
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
            "ai_host": "",
            "ai_api_key": "",
            "ai_model": "",
            "api_model": "",
            "mic_gain": 1.0,
            # New settings
            "outline_width": 0,
            "outline_color": "#000000",
            "custom_font": "",
        }

        if ARGOS_AVAILABLE:
            self.argos_translator = ArgosTranslator(logger=self.logger)

        self.display_thread = threading.Thread(
            target=self.update_display_state, daemon=True
        )
        self.display_thread.start()

    def is_repetitive_garbage(self, text):
        # Remove punctuation and split into words
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) > 10:
            word_counts = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            # If any single word repeats more than 5 times, it's likely garbage
            if max(word_counts.values()) > 5:
                self.logger.log(f"Filtered repetitive words: {text}", level="debug")
                return True

        # Check for long runs of same character (after removing spaces)
        cleaned = re.sub(r"[\s\.\,\!\?]", "", text)
        if len(cleaned) > 10:
            # Count consecutive identical characters
            max_run = 0
            current_run = 1
            for i in range(1, len(cleaned)):
                if cleaned[i] == cleaned[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            if max_run > 5:  # e.g., "aaaaaa"
                self.logger.log(f"Filtered character run: {text}", level="debug")
                return True

        return False

    def is_valid_transcription(self, text):
        # Reject if contains timestamp tokens like <|29.98|>
        if re.search(r"<\|.*?\|>", text):
            self.logger.log(
                f"Filtered transcription (contains timestamp): {text}", level="debug"
            )
            return False
        # Reject if too long (likely garbage repetition)
        if len(text) > 500:
            self.logger.log(f"Filtered transcription (too long): {text}", level="debug")
            return False

            # Reject repetitive garbage
        if self.is_repetitive_garbage(text):
            return False

        return True

    def load_language_models(self):
        models_dir = Path("vosk_models")
        if not models_dir.exists():
            return
        lang_patterns = {
            "en": ["en-us", "english"],
            "es": ["es", "spanish"],
            "fr": ["fr", "french"],
            "de": ["de", "german"],
            "it": ["it", "italian"],
            "pt": ["pt", "portuguese"],
        }
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_name = model_dir.name.lower()
            for lang_code, patterns in lang_patterns.items():
                if any(p in model_name for p in patterns):
                    self.available_vosk_models[lang_code] = str(model_dir)
                    break

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self.audio_queue.put(bytes(indata))

    def process_audio_hardware(self):
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=0.1)
                if self.settings["recognition_engine"] == "vosk":
                    self.vosk_audio_buffer.extend(data)
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text"):
                            text = result["text"]
                            print(f"[{self.session_hash[:8]}] FINAL: {text}")
                            self.logger.log(f"Recognized: {text}", level="info")
                            self.result_queue.put(("final", text))
                            self.last_update_time = time.time()
                            # Store the accumulated audio for whisper_translate
                            self.last_audio_chunk = bytes(self.vosk_audio_buffer)
                            # Clear buffer for next utterance
                            self.vosk_audio_buffer.clear()
                    else:
                        if self.settings["display_interim"]:
                            partial = json.loads(self.recognizer.PartialResult())
                            if partial.get("partial"):
                                self.result_queue.put(("interim", partial["partial"]))
                                self.last_update_time = time.time()
                else:
                    self.audio_buffer.extend(data)
                    self.buffer_duration += len(data) / (16000 * 2)
                    if self.buffer_duration >= self.target_buffer_duration:
                        self._process_whisper_buffer()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log(f"Audio error: {str(e)}", level="error")

    def _process_whisper_buffer(self):
        if not self.audio_buffer or not self.whisper_recognizer:
            return
        try:
            audio_bytes = bytes(self.audio_buffer)
            self.last_audio_chunk = audio_bytes
            transcription = self.whisper_recognizer.transcribe(
                audio_bytes, language=self.settings.get("whisper_language")
            )
            if transcription and not dots_or_stars(transcription):
                if self.is_valid_transcription(transcription):
                    self.result_queue.put(("final", transcription))
                    self.last_update_time = time.time()
                else:
                    self.logger.log(
                        "Discarded invalid Whisper transcription", level="info"
                    )
            self.audio_buffer.clear()
            self.buffer_duration = 0
        except Exception as e:
            self.logger.log(f"Whisper error: {str(e)}", level="error")
            self.audio_buffer.clear()
            self.buffer_duration = 0

    def start_recognition(self, model_path, microphone_index=None):
        try:
            if self.settings["recognition_engine"] == "vosk":
                if not model_path or not Path(model_path).exists():
                    msg = f"❌ Vosk model not found"
                    self.logger.log(msg, level="error")
                    return msg

                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, 16000)
                self.recognizer.SetWords(True)
                self.whisper_recognizer = None
            else:
                self.whisper_recognizer = WhisperRecognizer(
                    host=self.settings["whisper_host"],
                    api_key=self.settings.get("whisper_api_key"),
                    model=self.settings["whisper_model"],
                    logger=self.logger,
                )
                self.recognizer = None
                self.model = None

            self.is_running = True

            if (
                self.settings["audio_mode"] == "hardware"
                and microphone_index is not None
            ):
                self.settings["microphone"] = microphone_index
                device_info = sd.query_devices(microphone_index, "input")

                print(
                    f"[INFO] Using hardware mic [{self.settings['microphone']}]: {device_info['name']}"
                )
                self.logger.log(
                    f"Using hardware mic: {device_info['name']}", level="info"
                )

                self.stream = sd.RawInputStream(
                    samplerate=16000,
                    blocksize=8000,
                    device=microphone_index,
                    dtype="int16",
                    channels=1,
                    callback=self.audio_callback,
                )
                self.stream.start()
                self.process_thread = threading.Thread(
                    target=self.process_audio_hardware, daemon=True
                )
                self.process_thread.start()
                msg = "✅ Recognition started (Hardware)"
            else:
                msg = "✅ Recognition started (Browser)"
                # Start the processing thread even for browser
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
        except Exception as e:
            msg = f"❌ Error: {str(e)}"
            self.logger.log(msg, level="error")
            return msg

    def stop_recognition(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.audio_buffer.clear()
        self.vosk_audio_buffer.clear()
        msg = "Recognition stopped"
        self.logger.log(msg, level="success")
        return msg

    def _get_font_family_css(self):
        """Return the CSS font-family string, including custom font if selected."""
        if self.settings.get("custom_font"):
            # Create a safe font family name from filename
            font_name = Path(self.settings["custom_font"]).stem.replace(" ", "_")
            return f"'{font_name}', {self.settings['font_family']}"
        return self.settings["font_family"]

    def _get_font_face_css(self):
        """Return @font-face CSS for custom font, if any."""
        custom_font = self.settings.get("custom_font")
        if not custom_font:
            return ""
        font_name = Path(custom_font).stem.replace(" ", "_")
        return f"""
        @font-face {{
            font-family: '{font_name}';
            src: url('/fonts/{custom_font}') format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """

    def _get_outline_css(self, width, color):
        """Generate text-shadow outline with given width and color."""
        if width == 0:
            return ""
        # Use multiple shadows to create an outward outline
        offsets = []
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                if dx == 0 and dy == 0:
                    continue
                offsets.append(f"{dx}px {dy}px 0 {color}")
        return "text-shadow: " + ", ".join(offsets) + ";"

    def get_display_html(self, recognized_text, translated_text):
        alignment_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
        time_since = time.time() - self.last_update_time
        opacity = "0" if time_since > self.settings.get("fade_timeout", 5.0) else "1"

        # Base text style
        base_style = f"""
            font-family: {self._get_font_family_css()};
            transition: opacity 1s;
        """

        # Outline style
        outline_width = self.settings.get("outline_width", 0)
        outline_color = self.settings.get("outline_color", "#000000")
        outline_style = self._get_outline_css(outline_width, outline_color)

        # Font face for custom font (injected into the HTML)
        font_face = self._get_font_face_css()

        texts = []
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "before"
            and translated_text
        ):
            texts.append(
                f'<div style="font-size: {self.settings["translated_font_size"]}px; color: {self.settings["translated_color"]}; {base_style} {outline_style}">{translated_text}</div>'
            )
        texts.append(
            f'<div style="font-size: {self.settings["recognized_font_size"]}px; color: {self.settings["recognized_color"]}; {base_style} {outline_style}">{recognized_text}</div>'
        )
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "after"
            and translated_text
        ):
            texts.append(
                f'<div style="font-size: {self.settings["translated_font_size"]}px; color: {self.settings["translated_color"]}; {base_style} {outline_style}">{translated_text}</div>'
            )

        # Wrap everything in a container with the font-face style if needed
        return f'<style>{font_face}</style><div style="transition: opacity 1s; opacity: {opacity}; display: flex; flex-direction: column; align-items: {alignment_map[self.settings["text_alignment"]]}; justify-content: center; padding: 20px; background-color: {self.settings["background_color"]}; min-height: 200px;">{"".join(texts)}</div>'

    def update_display_state(self):
        while self.display_running:
            try:
                result_type, text = self.result_queue.get(timeout=0.1)
                if result_type == "stop":
                    break  # Exit on sentinel
                if result_type == "final":
                    self.current_recognized = text
                    if self.settings["enable_translation"]:
                        if (
                            self.settings.get("translation_mode") == "whisper"
                            and self.last_audio_chunk
                        ):
                            wt = WhisperRecognizer(
                                host=self.settings.get("whisper_translate_host"),
                                api_key=self.settings.get("whisper_translate_api_key"),
                                model=self.settings.get("whisper_translate_model"),
                                logger=self.logger,
                            )
                            self.current_translated = wt.translate(
                                self.last_audio_chunk
                            )
                        elif (
                            self.settings.get("translation_mode") == "argos"
                            and self.argos_translator
                        ):
                            self.current_translated = self.argos_translator.translate(
                                text,
                                self.settings.get("argos_source_lang"),
                                self.settings.get("argos_target_lang"),
                            )
                        elif self.translation_service:
                            self.current_translated = (
                                self.translation_service.translate(
                                    text,
                                    self.settings["source_language"].split("-")[0],
                                    self.settings["target_language"],
                                )
                            )
                        else:
                            self.current_translated = ""
                    else:
                        self.current_translated = ""
                elif result_type == "interim" and self.settings["display_interim"]:
                    self.current_recognized = text
            except queue.Empty:
                if not self.display_running:
                    break
                time.sleep(0.05)
            except Exception as e:
                if self.display_running:
                    self.logger.log(f"Display error: {str(e)}", level="error")
                time.sleep(0.1)

    def get_current_display(self):
        return (
            self.get_display_html(self.current_recognized, self.current_translated),
            self.current_recognized,
            self.current_translated if self.settings["enable_translation"] else "",
        )

    def update_logs(self):
        return self.logger.get_recent_logs(50)

    def generate_popout_html(self):
        alignment_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
        fade_ms = int(self.settings.get("fade_timeout", 5.0) * 1000)
        outline_width = self.settings.get("outline_width", 0)
        outline_color = self.settings.get("outline_color", "#000000")
        font_face = self._get_font_face_css()
        font_family = self._get_font_family_css()
        outline_style = self._get_outline_css(outline_width, outline_color)

        return f"""<!DOCTYPE html>
<html><head><title>Display</title><meta charset="UTF-8">
<style>
{font_face}
body,html{{margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;background:{self.settings["background_color"]};display:flex;align-items:center;justify-content:{alignment_map[self.settings["text_alignment"]]}}}
.container{{padding:20px;width:100%;text-align:{self.settings["text_alignment"]};transition:opacity 1s;opacity:1}}
.container.fade{{opacity:0}}
.rec{{font-size:{self.settings["recognized_font_size"]}px;color:{self.settings["recognized_color"]};margin:10px 0;font-family:{font_family};{outline_style}}}
.tra{{font-size:{self.settings["translated_font_size"]}px;color:{self.settings["translated_color"]};margin:10px 0;font-family:{font_family};{outline_style}}}
</style>
<script>let t=null;function reset(){{const c=document.getElementById('c');c.classList.remove('fade');if(t)clearTimeout(t);t=setTimeout(()=>c.classList.add('fade'),{fade_ms})}}async function update(){{try{{const r=await fetch('/popout_data/{self.popout_id}');const d=await r.json();const e=document.getElementById('r');const n=d.recognized||"Waiting...";if(n!==e.textContent){{e.textContent=n;reset()}}document.getElementById('t').textContent=d.translated||""}}catch(e){{}}}}setInterval(update,500);document.addEventListener('DOMContentLoaded',()=>{{update();reset()}});</script>
</head><body><div id="c" class="container"><div id="r" class="rec">Waiting...</div><div id="t" class="tra"></div></div></body></html>"""

    def force_vosk_cleanup(self):
        """Force cleanup of Vosk C++ objects"""
        if hasattr(self, "recognizer") and self.recognizer:
            try:
                # Force delete the recognizer
                del self.recognizer
            except:
                pass
            self.recognizer = None

        if hasattr(self, "model") and self.model:
            try:
                # Force delete the model
                del self.model
            except:
                pass
            self.model = None

        # Clear any cached models
        self.available_vosk_models.clear()

        # Force Python garbage collection
        import gc

        for i in range(3):  # Run multiple times
            gc.collect()

    def close(self):
        """Clean up ALL resources including Vosk model"""

        # Stop any running recognition
        if self.is_running:
            self.stop_recognition()

        # Stop the display thread
        self.display_running = False
        if self.display_thread and self.display_thread.is_alive():
            try:
                # Put a sentinel value to unblock the queue
                self.result_queue.put(("stop", ""))
                self.display_thread.join(timeout=2.0)
            except:
                pass

        # Clean up audio stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None

        # IMPORTANT: Clear Vosk model and recognizer
        # This helps Python garbage collection

        self.force_vosk_cleanup()
        self.recognizer = None
        self.model = None

        # Clear the audio buffer
        self.audio_buffer.clear()
        self.vosk_audio_buffer.clear()

        # Clear queues
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.audio_queue.get_nowait()
        except (queue.Empty, AttributeError):
            pass

        # If using whisper, clear that too
        self.whisper_recognizer = None

        # Clear argos translator if exists
        self.argos_translator = None

        # Force garbage collection
        import gc

        gc.collect()

        print(f"[SESSION CLEANUP] Completely closed session: {self.session_hash[:8]}")


def get_available_models():
    """Get available Vosk models"""
    models_dir = Path("vosk_models")
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        return []

    models = []
    for item in models_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            models.append((item.name, str(item)))
    return models


def get_microphones():
    """Get available hardware microphones"""
    try:
        devices = sd.query_devices()
        microphones = []
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                name = f"[ID:{i}] {device['name']}"
                microphones.append((name, i))
        return microphones
    except Exception as e:
        return [("Error getting devices", None)]


def get_or_create_app(session_hash):
    """Get existing app or create new one for this session"""
    with SESSION_LOCK:
        if session_hash not in SESSION_APPS:
            SESSION_APPS[session_hash] = VoiceTranslatorApp(session_hash)
            print(f"[NEW SESSION] {session_hash[:8]} | Total: {len(SESSION_APPS)}")
        return SESSION_APPS[session_hash]


def create_ui(args):
    with gr.Blocks(title="Voice Translator") as interface:
        session_info = gr.Markdown("### Session initializing...")

        gr.Markdown("# 🎤 Voice Translator")
        browser_session_data = gr.HTML(visible=True)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("🗣️ Speech Recognition", open=True):
                    recognition_engine = gr.Radio(
                        ["vosk", "whisper"],
                        value="vosk",
                        label="Recognition Engine",
                        info="Vosk: Offline, fast | Whisper: Online, more accurate",
                        interactive=True,
                    )

                    with gr.Group(visible=True) as vosk_settings:
                        vosk_models = get_available_models()
                        if not vosk_models:
                            vosk_choices = [("❌ No models detected", "")]
                            vosk_value = ""
                        else:
                            vosk_choices = vosk_models
                            vosk_value = vosk_models[0][1]

                        vosk_model_dropdown = gr.Dropdown(
                            choices=vosk_choices,
                            value=vosk_value,
                            label="Vosk Model",
                            info=f"Found {len(vosk_models)} models in vosk_models/"
                            if vosk_models
                            else "⚠️ No Vosk models found. Place models in vosk_models/",
                            interactive=bool(vosk_models),
                        )

                        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

                        display_interim = gr.Checkbox(
                            label="Show interim results",
                            value=True,
                            interactive=True,
                            info="Display partial recognition while speaking (VOSK only)",
                        )

                    with gr.Group(visible=False) as whisper_settings:
                        whisper_host = gr.Textbox(
                            label="Whisper API Host",
                            value="http://localhost:9000",
                            info="URL of your Whisper server",
                        )
                        whisper_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Leave empty if not required",
                        )
                        whisper_model = gr.Textbox(
                            label="Model",
                            value="whisper-large-v3",
                        )
                        whisper_language = gr.Textbox(
                            label="Language",
                            value="en",
                        )
                        test_whisper_btn = gr.Button("🔍 Test Connection", size="sm")

                with gr.Accordion("🎙️ Audio", open=True):
                    audio_mode = gr.Radio(
                        ["hardware", "browser"],
                        value="hardware",
                        label="Audio Mode",
                        info="Browser: Voice-activated | Hardware: System microphone",
                    )

                    with gr.Group(visible=True) as hardware_group:
                        gr.Markdown("*Hardware voice detection active*")
                        microphones = get_microphones()
                        mic_dropdown = gr.Dropdown(
                            choices=microphones,
                            label="Hardware Microphone",
                            value=microphones[0][1] if microphones else None,
                            info="Select system audio device",
                            interactive=True,
                        )

                        refresh_mic_btn = gr.Button("🔄 Refresh Devices", size="sm")

                    with gr.Group(visible=False) as browser_group:
                        gr.Markdown("*Browser voice detection active*")
                        # Status indicator
                        browser_status = gr.Textbox(
                            label="Browser Stream Status",
                            value="Not started",
                            interactive=False,
                            elem_id="browser-status",
                        )

                        mic_gain = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            label="Microphone Gain",
                            info="Amplify browser audio if input is too quiet.",
                            elem_id="browser-gain-slider",
                        )

                with gr.Accordion("🌐 Translation", open=True):
                    enable_translation = gr.Checkbox(
                        label="Enable Translation",
                        value=False,
                        info="Toggle translation on/off",
                    )

                    with gr.Group(visible=False) as translation_group:
                        translation_mode = gr.Radio(
                            [
                                "argos",
                                "ai",
                                "libretranslate",
                                "whisper",
                            ],
                            value="argos",
                            label="Translation Engine",
                            interactive=True,
                        )

                        with gr.Row():
                            source_lang = gr.Textbox(
                                label="From", value="en-US", scale=1, visible=False
                            )
                            target_lang = gr.Textbox(
                                label="To", value="es", scale=1, visible=False
                            )

                        with gr.Group(visible=False) as ai_settings:
                            ai_host = gr.Textbox(
                                label="API Host", value="http://localhost:11434/v1"
                            )
                            ai_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="Leave empty for Ollama",
                            )
                            ai_model = gr.Textbox(label="Model", value="llama3.2")

                        with gr.Group(visible=False) as libretranslate_settings:
                            libretranslate_host = gr.Textbox(
                                label="Host", value="http://localhost:5000"
                            )
                            libretranslate_api_key = gr.Textbox(
                                label="API Key", type="password"
                            )

                        # Whisper translate settings
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
                            gr.Markdown("*Translates audio to English using Whisper*")

                        # Argos translate settings
                        with gr.Group(visible=True) as argos_settings:
                            argos_source = gr.Textbox(
                                label="Source Language Code",
                                value="en",
                                info="2-letter code (en, es, fr, de, etc.)",
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
                                        "⚠️ **No language pairs installed.** Run `download_argos_models.py` to download models."
                                    )
                                else:
                                    gr.Markdown("✅ Argos is ready.")

            with gr.Column(scale=1):
                with gr.Row():
                    start_btn = gr.Button("▶️ Start", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop")

                status_text = gr.Textbox(label="Status", lines=2)

                gr.Markdown("### 📺 Display")
                popout_url = gr.Textbox(label="Popout URL", interactive=False)
                with gr.Group():
                    gr.Markdown(
                        "For a custom ID just enter the string and press 'Enter'"
                    )
                    custom_popout_id = gr.Textbox(
                        label="Custom Popout ID",
                        placeholder="Enter custom ID or leave empty for random",
                        interactive=True,
                        scale=3,
                    )
                    random_btn = gr.Button("🎲 Random", scale=1, size="sm")

                display_html = gr.HTML(value="<div>Loading...</div>")

                with gr.Accordion("Outputs", open=False):
                    recognized_output = gr.Textbox(label="Recognized")
                    translated_output = gr.Textbox(label="Translated")

                with gr.Accordion("🎨 Display Style", open=False):
                    # Combined font dropdown (system + custom)
                    custom_fonts = get_available_fonts()
                    font_choices = SYSTEM_FONTS + custom_fonts
                    font_selector = gr.Dropdown(
                        choices=font_choices,
                        value="Arial",
                        label="Font Family",
                        info="Select a system font or a custom font from the fonts/ folder.",
                        interactive=True,
                    )
                    refresh_fonts_btn = gr.Button("🔄 Refresh Fonts", size="sm")

                    recognized_font_size = gr.Slider(
                        12,
                        120,
                        48,
                        step=2,
                        label="Main text size",
                        interactive=True,
                    )
                    translated_font_size = gr.Slider(
                        12,
                        120,
                        32,
                        step=2,
                        label="Translation size",
                        interactive=True,
                    )
                    with gr.Row():
                        recognized_color = gr.ColorPicker(
                            label="Main color",
                            value="#FFFFFF",
                            interactive=True,
                        )
                        translated_color = gr.ColorPicker(
                            label="Translation color",
                            value="#CCCCCC",
                            interactive=True,
                        )
                        background_color = gr.ColorPicker(
                            label="Background",
                            value="#000000",
                            interactive=True,
                        )
                    text_alignment = gr.Radio(
                        ["left", "center", "right"],
                        value="center",
                        label="Align",
                        interactive=True,
                    )
                    translation_position = gr.Radio(
                        ["before", "after"],
                        value="after",
                        label="Translation position",
                        interactive=True,
                    )

                    # Fade timeout control
                    fade_timeout = gr.Slider(
                        minimum=0.1,
                        maximum=10.0,
                        value=5.0,
                        step=0.1,
                        label="Fade Timeout (seconds)",
                        info="Time before text fades out after last recognition",
                        interactive=True,
                    )

                    gr.Markdown("**Text Outline**")
                    outline_width = gr.Slider(
                        0, 10, 0, step=1, label="Outline width (px)", interactive=True
                    )
                    outline_color = gr.ColorPicker(
                        label="Outline color", value="#000000", interactive=True
                    )

        log_output = gr.Textbox(label="Log", lines=6)

        def initialize_all_settings(request: gr.Request):
            """Initialize all settings from UI defaults when app loads"""
            app = get_or_create_app(request.session_hash)

            # Get available models and microphones
            models = get_available_models()
            mics = get_microphones()

            # Set model path - use first available model or stored value
            if models:
                if (
                    not app.settings["vosk_model"]
                    or not Path(app.settings["vosk_model"]).exists()
                ):
                    app.settings["vosk_model"] = models[0][1]

            # Set microphone
            if mics:
                if app.settings["microphone"] not in [m[1] for m in mics]:
                    app.settings["microphone"] = mics[0][1]

            # Log the initialization
            app.logger.log(
                f"Initialized settings for session {request.session_hash[:8]}",
                level="info",
            )
            return app

        # Event handlers - ALL need request parameter
        def update_recognition_engine(engine, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["recognition_engine"] = engine
            return {
                vosk_settings: gr.update(visible=(engine == "vosk")),
                whisper_settings: gr.update(visible=(engine == "whisper")),
            }

        def update_audio_mode(mode, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["audio_mode"] = mode
            return {
                hardware_group: gr.update(visible=(mode == "hardware")),
                browser_group: gr.update(visible=(mode == "browser")),
            }

        def update_translation_mode(mode, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["translation_mode"] = mode

            return {
                ai_settings: gr.update(visible=(mode == "ai")),
                libretranslate_settings: gr.update(visible=(mode == "libretranslate")),
                whisper_translate_settings: gr.update(visible=(mode == "whisper")),
                argos_settings: gr.update(visible=(mode == "argos")),
                source_lang: gr.update(visible=(mode != "argos")),
                target_lang: gr.update(visible=(mode != "argos")),
            }

        def update_translation_toggle(enabled, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["enable_translation"] = enabled
            if not enabled:
                app.current_translated = ""
            return gr.update(visible=enabled)

        def update_display_interim_toggle(enabled, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["display_interim"] = enabled

        def update_source_lang(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["source_language"] = value

        def update_target_lang(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["target_language"] = value

        def update_ai_host(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["ai_host"] = value

        def update_ai_api_key(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["ai_api_key"] = value

        def update_ai_model(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["ai_model"] = value

        def update_libretranslate_host(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["libretranslate_host"] = value

        def update_libretranslate_api_key(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["libretranslate_api_key"] = value

        def update_font_selector(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            # value is either a system font name or a custom font filename
            # Check if it's a custom font (exists in fonts/)
            custom_fonts = [f[1] for f in get_available_fonts()]
            if value in custom_fonts:
                app.settings["custom_font"] = value
                # Also set a fallback system font (maybe keep previous)
                # We'll keep the previous font_family as fallback
            else:
                app.settings["font_family"] = value
                app.settings["custom_font"] = ""

        def update_recognized_font_size(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["recognized_font_size"] = value

        def update_translated_font_size(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["translated_font_size"] = value

        def update_recognized_color(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["recognized_color"] = value

        def update_translated_color(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["translated_color"] = value

        def update_background_color(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["background_color"] = value

        def update_text_alignment(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["text_alignment"] = value

        def update_translation_position(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["translation_position"] = value

        def update_whisper_host(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_host"] = value

        def update_whisper_api_key(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_api_key"] = value

        def update_whisper_model(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_model"] = value

        def update_whisper_language(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_language"] = value

        def update_whisper_trans_host(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_translate_host"] = value

        def update_whisper_trans_api_key(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_translate_api_key"] = value

        def update_whisper_trans_model(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["whisper_translate_model"] = value

        def update_argos_source(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["argos_source_lang"] = value

        def update_argos_target(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["argos_target_lang"] = value

        def update_fade_timeout(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["fade_timeout"] = value

        def update_mic_gain(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["mic_gain"] = value

        def update_custom_popout_id(value, request: gr.Request):
            """Update the popout ID when user types a custom value."""
            app = get_or_create_app(request.session_hash)
            if value and value.strip():
                # Basic sanitization: allow alphanumeric, hyphen, underscore
                sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", value.strip())
                if sanitized:
                    app.popout_id = sanitized
                else:
                    app.popout_id = secrets.token_urlsafe(16)
            else:
                # If empty, generate a random one
                app.popout_id = secrets.token_urlsafe(16)
            # Return updated popout URL
            return (
                f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                app.popout_id,
            )

        def generate_random_popout(value, request: gr.Request):
            """Generate a new random popout ID."""
            app = get_or_create_app(request.session_hash)
            app.popout_id = secrets.token_urlsafe(16)
            return (
                f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                app.popout_id,
            )

        # New update functions
        def update_outline_width(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["outline_width"] = value

        def update_outline_color(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["outline_color"] = value

        def start_rec(request: gr.Request):
            app = get_or_create_app(request.session_hash)
            model = (
                app.settings["vosk_model"]
                if app.settings["recognition_engine"] == "vosk"
                else None
            )
            mic = app.settings["microphone"]
            return app.start_recognition(model, mic)

        def stop_rec(request: gr.Request):
            return get_or_create_app(request.session_hash).stop_recognition()

        def update_mic(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["microphone"] = value

        def update_model(value, request: gr.Request):
            app = get_or_create_app(request.session_hash)
            app.settings["vosk_model"] = value

        def test_whisper_connection(request: gr.Request):
            """Test connection to Whisper API"""
            app = get_or_create_app(request.session_hash)
            try:
                host = app.settings["whisper_host"]
                api_key = (
                    app.settings["whisper_api_key"]
                    if app.settings["whisper_api_key"]
                    else None
                )

                if not host:
                    return "❌ Please enter a Whisper API host URL"

                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                response = requests.get(host, headers=headers, timeout=5)

                if response.status_code == 200:
                    return f"✅ Connected to Whisper API at {host}"
                else:
                    for endpoint in ["/health", "/models", "/audio/transcriptions"]:
                        try:
                            test_url = host.rstrip("/") + endpoint
                            response = requests.get(
                                test_url, headers=headers, timeout=5
                            )
                            if response.status_code == 200:
                                return f"✅ Connected to Whisper API at {test_url}"
                        except:
                            continue

                    test_url = host.rstrip("/") + "/audio/transcriptions"
                    response = requests.post(test_url, headers=headers, timeout=5)
                    if response.status_code != 405:
                        return f"✅ Whisper API endpoint available at {test_url}"
                    else:
                        return "⚠️ Whisper API responds but method check failed. API might be working."

            except requests.exceptions.ConnectionError:
                return f"❌ Cannot connect to {host}. Make sure the Whisper server is running."
            except Exception as e:
                return f"❌ Error testing connection: {str(e)}"

        def update_display(request: gr.Request):
            return get_or_create_app(request.session_hash).get_current_display()

        def update_logs(request: gr.Request):
            return get_or_create_app(request.session_hash).update_logs()

        def update_browser_status(status_msg):
            return gr.update(value=status_msg)

        def cleanup_user_data(request: gr.Request):
            """Clean up the session data when the session ends."""
            session_hash = request.session_hash

            with SESSION_LOCK:
                if session_hash in SESSION_APPS:
                    app = SESSION_APPS[session_hash]
                    # Clean up the app resources
                    app.close()
                    # Remove from session storage
                    del SESSION_APPS[session_hash]
                    print(
                        f"[SESSION CLEANUP] Removed session {session_hash[:8]}. Active: {len(SESSION_APPS)}"
                    )
                else:
                    print(f"[SESSION CLEANUP] Session {session_hash[:8]} not found.")

        def handle_ui_load(request: gr.Request):
            """Handle UI load by initializing all settings and component values"""
            app = get_or_create_app(request.session_hash)
            # Initialize all settings if not already done
            initialize_all_settings(request)

            # Get all current values from app settings
            settings = app.settings

            # Determine font dropdown value
            if settings.get("custom_font"):
                font_value = settings["custom_font"]
            else:
                font_value = settings["font_family"]

            # Generate session data HTML for JS
            ws_url = f"ws://{args.host}:{args.port}/ws/{request.session_hash}"
            session_html = f'<div id="session-data" data-session="{request.session_hash}" data-ws-url="{ws_url}"><b>Each tab = separate session • Refresh = new session</b></div>'

            # Return a dictionary of updates for all components
            return {
                session_info: f"### 🎯 Session: `{request.session_hash[:8]}` | Active: {len(SESSION_APPS)}",
                popout_url: f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                vosk_model_dropdown: settings["vosk_model"],
                mic_dropdown: settings["microphone"],
                recognition_engine: settings["recognition_engine"],
                audio_mode: settings["audio_mode"],
                enable_translation: settings["enable_translation"],
                display_interim: settings["display_interim"],
                translation_mode: settings["translation_mode"],
                source_lang: settings["source_language"],
                target_lang: settings["target_language"],
                font_selector: font_value,
                recognized_font_size: settings["recognized_font_size"],
                translated_font_size: settings["translated_font_size"],
                recognized_color: settings["recognized_color"],
                translated_color: settings["translated_color"],
                background_color: settings["background_color"],
                text_alignment: settings["text_alignment"],
                translation_position: settings["translation_position"],
                whisper_host: settings["whisper_host"],
                whisper_api_key: settings["whisper_api_key"],
                whisper_model: settings["whisper_model"],
                whisper_language: settings["whisper_language"],
                whisper_trans_host: settings["whisper_translate_host"],
                whisper_trans_api_key: settings["whisper_translate_api_key"],
                whisper_trans_model: settings["whisper_translate_model"],
                argos_source: settings["argos_source_lang"],
                argos_target: settings["argos_target_lang"],
                fade_timeout: settings["fade_timeout"],
                libretranslate_host: settings["libretranslate_host"],
                libretranslate_api_key: settings["libretranslate_api_key"],
                ai_host: settings["ai_host"],
                ai_api_key: settings["ai_api_key"],
                ai_model: settings["ai_model"],
                mic_gain: settings["mic_gain"],
                outline_width: settings["outline_width"],
                outline_color: settings["outline_color"],
                browser_session_data: session_html,
                browser_status: "Ready",
                custom_popout_id: app.popout_id,
            }

        # Wire up events
        recognition_engine.change(
            update_recognition_engine,
            [recognition_engine],
            [vosk_settings, whisper_settings],
        )

        audio_mode.change(
            update_audio_mode, [audio_mode], [hardware_group, browser_group]
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

        custom_popout_id.submit(
            update_custom_popout_id, [custom_popout_id], [popout_url, custom_popout_id]
        )
        random_btn.click(
            generate_random_popout, [random_btn], [popout_url, custom_popout_id]
        )

        enable_translation.change(
            update_translation_toggle, [enable_translation], [translation_group]
        )

        display_interim.change(update_display_interim_toggle, [display_interim])

        source_lang.change(update_source_lang, [source_lang])
        target_lang.change(update_target_lang, [target_lang])
        ai_host.change(update_ai_host, [ai_host])
        ai_api_key.change(update_ai_api_key, [ai_api_key])
        ai_model.change(update_ai_model, [ai_model])
        libretranslate_host.change(update_libretranslate_host, [libretranslate_host])
        libretranslate_api_key.change(
            update_libretranslate_api_key, [libretranslate_api_key]
        )
        font_selector.change(update_font_selector, [font_selector])
        recognized_font_size.change(update_recognized_font_size, [recognized_font_size])
        translated_font_size.change(update_translated_font_size, [translated_font_size])
        recognized_color.change(update_recognized_color, [recognized_color])
        translated_color.change(update_translated_color, [translated_color])
        background_color.change(update_background_color, [background_color])
        text_alignment.change(update_text_alignment, [text_alignment])
        translation_position.change(update_translation_position, [translation_position])
        whisper_host.change(update_whisper_host, [whisper_host])
        whisper_api_key.change(update_whisper_api_key, [whisper_api_key])
        whisper_model.change(update_whisper_model, [whisper_model])
        whisper_language.change(update_whisper_language, [whisper_language])
        whisper_trans_host.change(update_whisper_trans_host, [whisper_trans_host])
        whisper_trans_api_key.change(
            update_whisper_trans_api_key, [whisper_trans_api_key]
        )
        whisper_trans_model.change(update_whisper_trans_model, [whisper_trans_model])
        argos_source.change(update_argos_source, [argos_source])
        argos_target.change(update_argos_target, [argos_target])
        fade_timeout.change(update_fade_timeout, [fade_timeout])

        mic_gain.change(update_mic_gain, [mic_gain])

        outline_width.change(update_outline_width, [outline_width])
        outline_color.change(update_outline_color, [outline_color])

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

        test_whisper_btn.click(test_whisper_connection, outputs=[status_text])

        # Start and stop buttons with JS callbacks
        start_btn.click(
            fn=start_rec,
            outputs=[status_text],
            js="startBrowserStreaming",
        ).then(fn=lambda: "Streaming started", outputs=[browser_status])

        stop_btn.click(
            fn=stop_rec,
            outputs=[status_text],
            js="stopBrowserStreaming",
        ).then(fn=lambda: "Streaming stopped", outputs=[browser_status])

        mic_dropdown.change(update_mic, [mic_dropdown])
        vosk_model_dropdown.change(update_model, [vosk_model_dropdown])

        timer = gr.Timer(0.2)
        timer.tick(
            update_display, outputs=[display_html, recognized_output, translated_output]
        )
        log_timer = gr.Timer(1)
        log_timer.tick(update_logs, outputs=[log_output])

        interface.unload(cleanup_user_data)

        interface.load(
            handle_ui_load,
            inputs=None,
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
                translation_position,
                whisper_host,
                whisper_api_key,
                whisper_model,
                whisper_language,
                whisper_trans_host,
                whisper_trans_api_key,
                whisper_trans_model,
                argos_source,
                argos_target,
                fade_timeout,
                libretranslate_host,
                libretranslate_api_key,
                ai_host,
                ai_api_key,
                ai_model,
                mic_gain,
                outline_width,
                outline_color,
                browser_session_data,
                browser_status,
                custom_popout_id,
            ],
        )

    return interface


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


def cleanup_old_sessions(timeout_minutes=1):
    """Periodically clean up old sessions"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        current_time = time.time()
        with SESSION_LOCK:
            sessions_to_remove = []

            for session_hash, app in SESSION_APPS.items():
                # Check if session hasn't been updated for timeout period
                time_since_last_update = current_time - app.last_update_time
                if time_since_last_update > (timeout_minutes * 60):
                    sessions_to_remove.append(session_hash)

            for session_hash in sessions_to_remove:
                app = SESSION_APPS[session_hash]
                app.close()
                del SESSION_APPS[session_hash]
                print(f"[AUTO CLEANUP] Removed inactive session {session_hash[:8]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    # Start session cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()

    args = parser.parse_args()

    fastapi_app = FastAPI()

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.websocket("/ws/{session_hash}")
    async def websocket_endpoint(websocket: WebSocket, session_hash: str):
        await websocket.accept()
        with SESSION_LOCK:
            app = SESSION_APPS.get(session_hash)
        if not app:
            await websocket.close(code=1008, reason="Session not found")
            return
        try:
            while True:
                data = await websocket.receive_bytes()
                # Data is expected to be raw PCM int16, mono, 16kHz
                app.audio_queue.put(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket error for session {session_hash}: {e}")

    @fastapi_app.get("/popout/{popout_id}")
    async def get_popout(popout_id: str):
        with SESSION_LOCK:
            for app in SESSION_APPS.values():
                if app.popout_id == popout_id:
                    return HTMLResponse(content=app.generate_popout_html())
        return HTMLResponse("<h1>Not found</h1>", 404)

    @fastapi_app.get("/popout_data/{popout_id}")
    async def get_data(popout_id: str):
        with SESSION_LOCK:
            for app in SESSION_APPS.values():
                if app.popout_id == popout_id:
                    html, rec, trans = app.get_current_display()
                    return JSONResponse(
                        {
                            "recognized": rec,
                            "translated": trans
                            if app.settings["enable_translation"]
                            else "",
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
    fastapi_app = gr.mount_gradio_app(fastapi_app, interface, path="/", head=js)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🎤 Voice Translator - Multi-Session                 ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                                  ║
║                                                              ║
║  ✨ Each tab/refresh = new independent session              ║
║  🎙️ Browser mode uses Web Audio API + WebSocket             ║
║  🎨 Added text outline & custom font support                ║
║  🔒 Thread-safe session management                           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(fastapi_app, host=args.host, port=args.port, log_config=LOG_CONFIG)
