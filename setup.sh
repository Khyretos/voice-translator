#!/bin/bash
# Quick start script for Voice Translator

set -e

echo "🎤 Voice Translator - Quick Start Setup"
echo "========================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.8+"; exit 1; }

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Warn about Python 3.14
if [[ "$PYTHON_VERSION" == "3.14" ]]; then
    echo "⚠️  Warning: Python 3.14 detected. Some packages may have compatibility issues."
    echo "   If installation fails, consider using Python 3.11 or 3.12."
    echo ""
fi
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
echo ""

# Install packages with error handling
echo "Installing core dependencies..."
if ! pip install -r requirements.txt; then
    echo ""
    echo "⚠️  Some packages failed to install. Trying alternative installation..."
    echo ""
    
    # Install packages one by one with error handling
    pip install vosk --break-system-packages 2>/dev/null || pip install vosk || echo "⚠️  vosk installation had issues"
    pip install sounddevice --break-system-packages 2>/dev/null || pip install sounddevice || echo "⚠️  sounddevice installation had issues"
    pip install "numpy<2.0" --break-system-packages 2>/dev/null || pip install "numpy<2.0" || echo "⚠️  numpy installation had issues"
    pip install requests --break-system-packages 2>/dev/null || pip install requests || echo "⚠️  requests installation had issues"
    pip install translators --break-system-packages 2>/dev/null || pip install translators || echo "⚠️  translators installation had issues"
    
    # Try gradio without optional dependencies if it fails
    if ! pip install "gradio>=4.44.0" --break-system-packages 2>/dev/null; then
        echo "Installing gradio without optional dependencies..."
        pip install gradio --no-deps --break-system-packages 2>/dev/null || pip install gradio --no-deps
        pip install aiofiles altair fastapi ffmpy gradio-client httpx huggingface-hub jinja2 markupsafe matplotlib numpy orjson packaging pandas pillow pydantic pydub python-multipart pyyaml ruff semantic-version tomlkit typer typing-extensions uvicorn websockets --break-system-packages 2>/dev/null || true
    fi
fi

echo "✅ Dependencies installed"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p models logs
echo "✅ Directories created"
echo ""

# Check for models
echo "🔍 Checking for Vosk models..."
if [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "⚠️  No models found in models/ directory"
    echo ""
    echo "Would you like to download a small English model? (40 MB)"
    read -p "Download model? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python download_models.py en-us-small
    else
        echo "ℹ️  You can download models manually:"
        echo "   python download_models.py en-us-small"
        echo ""
        echo "   Or visit: https://alphacephei.com/vosk/models"
    fi
else
    echo "✅ Models found:"
    ls -1 models/
fi
echo ""

echo "✨ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🐳 Or using Docker:"
echo "   docker-compose up -d"
echo ""
echo "📚 For more information, see README.md"
