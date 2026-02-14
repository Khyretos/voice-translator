@echo off
REM Voice Translator - Windows Start Script

echo Starting Voice Translator...
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Start the application
python app.py

pause
