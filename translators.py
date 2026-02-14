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
            else:  # internal
                return self._translate_internal(text, source_lang, target_lang)
        except Exception as e:
            self.logger.log(f"Translation error ({mode}): {str(e)}", level="error")
            return f"[Translation error: {str(e)}]"

    def _translate_internal(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using internal translators library"""
        try:
            if ts is None:
                return "[translators library not installed]"

            # Use Google Translate as default
            result = ts.translate_text(
                text,
                translator="google",
                from_language=source_lang,
                to_language=target_lang,
            )

            self.logger.log(
                f"Internal translation: '{text}' -> '{result}' ({source_lang}->{target_lang})",
                level="info",
            )

            return result

        except Exception as e:
            self.logger.log(f"Internal translation error: {str(e)}", level="error")
            raise

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
            host = self.settings.get("libretranslate_host", "http://localhost:5000")
            api_key = self.settings.get("libretranslate_api_key", "")

            # Ensure host has correct endpoint
            if not host.endswith("/translate"):
                if host.endswith("/"):
                    host = f"{host}translate"
                else:
                    host = f"{host}/translate"

            payload = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text",
            }

            if api_key:
                payload["api_key"] = api_key

            self.logger.log(f"LibreTranslate request to {host}", level="info")

            response = requests.post(host, json=payload, timeout=10)

            response.raise_for_status()
            result = response.json()

            translated_text = result.get("translatedText", "")

            self.logger.log(
                f"LibreTranslate translation: '{text}' -> '{translated_text}' ({source_lang}->{target_lang})",
                level="info",
            )

            return translated_text

        except requests.exceptions.RequestException as e:
            self.logger.log(
                f"LibreTranslate API request error: {str(e)}", level="error"
            )
            raise
        except (KeyError, ValueError) as e:
            self.logger.log(
                f"LibreTranslate API response parsing error: {str(e)}", level="error"
            )
            raise
