FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY translators.py .
COPY logger.py .

# Create directories
RUN mkdir -p /app/vosk_models /app/argos-models /app/logs

# Expose Gradio default port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run the application
ENTRYPOINT ["python", "app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
