FROM python:3.11-slim

# Install system dependencies + build tools (required for webrtcvad)
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    libsndfile1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to slim down the image
RUN apt-get remove -y build-essential python3-dev && apt-get autoremove -y

# Copy application files (multiple sources -> destination must end with /)
# The app is split across several modules (see REFACTOR.md) and all of them
# are needed at startup — copy every top-level .py instead of listing them,
# so a newly added module (e.g. live_whisper.py) can't be forgotten here.
COPY *.py ./

# Create required directories
RUN mkdir -p /voice_translator/vosk_models /voice_translator/argos_models /voice_translator/logs /voice_translator/fonts /app/settings

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run the application with host/port arguments
CMD ["python", "voice_translator.py", "--host", "0.0.0.0", "--port", "7860"]