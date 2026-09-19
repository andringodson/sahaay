@echo off
REM Launch Sahaay. Any arguments are passed straight through, so:
REM   run.bat            -> web UI
REM   run.bat --mock     -> no models needed, scripted transcript
REM   run.bat --cli      -> captions in this window
REM   run.bat --device   -> print the execution provider and exit

setlocal
set "HERE=%~dp0"
set "VENV_PY=%HERE%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo   Sahaay is not set up yet. Run this first:
    echo.
    echo     powershell -ExecutionPolicy Bypass -File install.ps1
    echo.
    exit /b 1
)

"%VENV_PY%" -m sahaay %*
endlocal
