import requests

try:
    import translators as ts
except ImportError:
    ts = None


class TranslationService:
    """Handles translation using different backends"""

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the configured translation mode"""
        if not text or not text.strip():
            return ""

        mode = self.settings.get("translation_mode", "internal")

        try:
            if mode == "ai":
                return self._translate_ai(text, source_lang, target_lang)
            elif mode == "libretranslate":
                return self._translate_libretranslate(text, source_lang, target_lang)
        except Exception as e:
            self.logger.log(f"Translation error ({mode}): {str(e)}", level="error")
            return f"[Translation error: {str(e)}]"

    def _translate_ai(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using AI service (OpenAI-compatible)"""
        try:
            host = self.settings.get("ai_host", "http://localhost:11434/v1")
            api_key = self.settings.get("ai_api_key", "")
            model = self.settings.get("ai_model", "llama3.2")

            # Ensure host ends with correct endpoint
            if not host.endswith("/chat/completions"):
                if host.endswith("/v1"):
                    host = f"{host}/chat/completions"
                elif host.endswith("/"):
                    host = f"{host}v1/chat/completions"
                else:
                    host = f"{host}/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Create translation prompt
            prompt = f"Translate the following text from {source_lang} to {target_lang}. Only provide the translation, no explanations if you cannot translate it return a single space:\n\n{text}"

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            }

            self.logger.log(
                f"AI translation request to {host} with model {model}", level="info"
            )

            response = requests.post(host, headers=headers, json=payload, timeout=10)

            response.raise_for_status()
            result = response.json()

            translated_text = result["choices"][0]["message"]["content"].strip()

            self.logger.log(
                f"AI translation: '{text}' -> '{translated_text}' ({source_lang}->{target_lang})",
                level="info",
            )

            return translated_text

        except requests.exceptions.RequestException as e:
            self.logger.log(f"AI API request error: {str(e)}", level="error")
            raise
        except (KeyError, IndexError) as e:
            self.logger.log(f"AI API response parsing error: {str(e)}", level="error")
            raise

    def _translate_libretranslate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate using LibreTranslate service"""
        try:
            # Normalize language codes: "en-US" -> "en", "es-ES" -> "es", etc.
            src = source_lang.split("-")[0] if source_lang else "auto"
            tgt = target_lang.split("-")[0] if target_lang else "en"

            host = self.settings.get("libretranslate_host", "http://localhost:5000")
            api_key = self.settings.get("libretranslate_api_key", "")

            # Build URL: ensure it ends with /translate
            base = host.rstrip("/")
            if not base.endswith("/translate"):
                url = f"{base}/translate"
            else:
                url = base

            payload = {"q": text, "source": src, "target": tgt, "format": "text"}
            if api_key:
                payload["api_key"] = api_key

            self.logger.log(
                f"LibreTranslate request to {url} ({src}->{tgt})", level="info"
            )

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            translated_text = result.get("translatedText", "")

            if not translated_text:
                self.logger.log(
                    "LibreTranslate returned empty translation", level="warning"
                )
                return ""

            self.logger.log(
                f"LibreTranslate: '{text}' -> '{translated_text}' ({src}->{tgt})",
                level="info",
            )
            return translated_text

        except requests.exceptions.ConnectionError:
            self.logger.log(
                "LibreTranslate connection error - is the server running?",
                level="error",
            )
            return f"[LibreTranslate not reachable at {host}]"
        except requests.exceptions.Timeout:
            self.logger.log("LibreTranslate request timed out", level="error")
            return "[LibreTranslate timeout]"
        except Exception as e:
            self.logger.log(f"LibreTranslate error: {str(e)}", level="error")
            raise
