@echo off
REM Voice Translator - Windows Quick Start Script

echo ========================================
echo Voice Translator - Quick Start Setup
echo ========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Warn about Python 3.14
echo %PYTHON_VERSION% | findstr /C:"3.14" >nul
if not errorlevel 1 (
    echo.
    echo [WARNING] Python 3.14 detected. Some packages may have compatibility issues.
    echo           If installation fails, consider using Python 3.11 or 3.12.
)
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
echo.

echo Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install. Trying alternative installation...
    echo.
    
    REM Install packages one by one
    pip install vosk
    pip install sounddevice
    pip install "numpy<2.0"
    pip install requests
    pip install translators
    pip install "gradio>=4.44.0"
    pip install argostranslate
)

echo [OK] Dependencies installed
echo.

REM Create directories
echo Creating directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
echo [OK] Directories created
echo.

REM Check for models
echo Checking for Vosk models...
dir /b models 2>nul | findstr /r "." >nul
if errorlevel 1 (
    echo [WARNING] No models found in models\ directory
    echo.
    echo Would you like to download a small English model? (40 MB)
    set /p choice="Download model? (y/n): "
    if /i "%choice%"=="y" (
        python download_models.py en-us-small
    ) else (
        echo.
        echo You can download models manually:
        echo    python download_models.py en-us-small
        echo.
        echo Or visit: https://alphacephei.com/vosk/models
    )
) else (
    echo [OK] Models found:
    dir /b models
)
echo.

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the application:
echo    venv\Scripts\activate.bat
echo    python app.py
echo.
echo Or simply run: start.bat
echo.
echo For more information, see README.md
echo.
pause
