import logging
from datetime import datetime
from pathlib import Path
from collections import deque


class Logger:
    def __init__(self, session_id=None):
        self.session_id = session_id or "default"
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # In-memory log buffer for UI display (per session)
        self.log_buffer = deque(maxlen=100)

        # File logger
        log_filename = (
            self.log_dir
            / f"voice_translator_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_id[:8]}.log"
        )

        self.logger = logging.getLogger(f"VoiceTranslator_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_filename, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            f"[{self.session_id[:8]}] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.log(f"Logger initialized for session {self.session_id[:8]}", level="info")

    def log(self, message, level="info"):
        """Log a message and add to buffer"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add to in-memory buffer for UI
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.log_buffer.append(log_entry)

        # Log to file
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "success":
            self.logger.info(f"✓ {message}")
        else:
            self.logger.info(message)

    def get_recent_logs(self, count=50):
        """Get recent logs for UI display"""
        logs = list(self.log_buffer)
        return "\n".join(logs[-count:]) if logs else "No logs yet"
