@echo off
title Meeting Transcriber ^& Analyser

echo.
echo  ================================================
echo   Meeting Transcriber ^& Analyser
echo   Milestone 1 ^& 2
echo  ================================================
echo.

:: ── Check that the virtual environment exists ─────────────────────────────
if not exist "%~dp0myenv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo.
    echo  Run this first:
    echo    python -m venv myenv
    echo    myenv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Check that streamlit is installed inside the venv ────────────────────
if not exist "%~dp0myenv\Scripts\streamlit.exe" (
    echo  [ERROR] Streamlit not found in the virtual environment.
    echo.
    echo  Run this first:
    echo    myenv\Scripts\pip.exe install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Check for .env file ───────────────────────────────────────────────────
if not exist "%~dp0.env" (
    echo  [WARNING] .env file not found.
    echo  The AI summary feature requires GEMINI_API_KEY.
    echo.
    echo  To set it up:
    echo    copy .env.example .env
    echo    Then edit .env and add your key.
    echo.
    echo  The app will still run — Python analysis works without a key.
    echo.
)

:: ── Start the app ─────────────────────────────────────────────────────────
echo  Starting Streamlit...
echo  Open your browser at: http://localhost:8501
echo.
echo  Press Ctrl+C to stop the app.
echo.

"%~dp0myenv\Scripts\streamlit.exe" run "%~dp0app.py"

:: ── Reached here means the app exited ────────────────────────────────────
echo.
echo  App stopped.
pause
