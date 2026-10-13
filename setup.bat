@echo off
REM Voice Translator - Windows Quick Start Script

echo ========================================
echo Voice Translator - Quick Start Setup
echo ========================================
echo.

REM Check Python (prefer 3.12: torch, onnxruntime etc. have no wheels for newer versions)
echo Checking Python installation...
set "PY=python"
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('%PY% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Warn about Python 3.13 and newer
echo %PYTHON_VERSION% | findstr /R /C:"^3\.1[3-9]" >nul
if not errorlevel 1 (
    echo.
    echo [WARNING] Python %PYTHON_VERSION% detected. torch, onnxruntime and others have no wheels yet.
    echo           Install Python 3.12 ^(py -3.12 is used automatically when present^).
    if not "%VT_FORCE_PYTHON%"=="1" (
        echo [ERROR] Stopping: this Python has no wheels for the dependencies. Set VT_FORCE_PYTHON=1 to try anyway.
        pause
        exit /b 1
    )
)
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    %PY% -m venv venv
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
set "PIP_FAILED=0"
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install. Trying alternative installation...
    echo.
    
    pip install vosk
    if errorlevel 1 set "PIP_FAILED=1"
    pip install sounddevice
    if errorlevel 1 set "PIP_FAILED=1"
    pip install "numpy<2.0"
    if errorlevel 1 set "PIP_FAILED=1"
    pip install requests
    if errorlevel 1 set "PIP_FAILED=1"
    pip install translators
    if errorlevel 1 set "PIP_FAILED=1"
    pip install "gradio>=4.44.0"
    if errorlevel 1 set "PIP_FAILED=1"
    pip install argostranslate
    if errorlevel 1 set "PIP_FAILED=1"
    pip install webrtcvad-wheels
    if errorlevel 1 set "PIP_FAILED=1"
    pip install websockets
    if errorlevel 1 set "PIP_FAILED=1"
    pip install moonshine-voice
    if errorlevel 1 set "PIP_FAILED=1"
    pip install tokenizers
    if errorlevel 1 set "PIP_FAILED=1"
    pip install huggingface_hub
    if errorlevel 1 set "PIP_FAILED=1"
)

if "%PIP_FAILED%"=="1" (
    echo.
    echo [ERROR] Dependencies could not be installed. See the pip output above.
    pause
    exit /b 1
)

echo [OK] Dependencies installed.
echo.

REM Create directories
echo Creating directories...
if not exist "logs" mkdir logs
echo [OK] Directories created
echo.

REM Check for models
echo Checking for Vosk models...
dir /b vosk_models 2>nul | findstr /r "." >nul
if not errorlevel 1 goto models_found
echo [WARNING] No models found in vosk_models\ directory
echo.
if /i "%VT_DOWNLOAD_MODEL%"=="y" goto download_model
echo Would you like to download a small English model? ^(40 MB^)
choice /c yn /t 30 /d n /m "Download model" 2>nul
if errorlevel 255 goto no_console
if errorlevel 2 goto skip_model
:download_model
python download_vosk_models.py --lang en-us --small
if errorlevel 1 echo [WARNING] Model download failed. Retry later with: python download_vosk_models.py --lang en-us --small
goto after_models
:no_console
echo [INFO] No console, skipping model download.
:skip_model
echo.
echo You can download models manually:
echo    python download_vosk_models.py --lang en-us --small
echo.
echo Or visit: https://alphacephei.com/vosk/models
goto after_models
:models_found
echo [OK] Models found:
dir /b vosk_models
:after_models
echo.

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the application:
echo    venv\Scripts\activate.bat
echo    python voice_translator.py
echo.
echo Or simply run: start.bat
echo.
echo For more information, see README.md
echo.
pause
