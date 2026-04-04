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
COPY app.py translators.py logger.py ./

# Create required directories
RUN mkdir -p /app/vosk_models /app/argos_models /app/logs /app/fonts

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run the application with host/port arguments
CMD ["python", "voice_translator.py", "--host", "0.0.0.0", "--port", "7860"]