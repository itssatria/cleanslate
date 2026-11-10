@echo off
REM One-click launcher for Windows: creates a venv on first run, then starts the app.
if not exist .venv (
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -e ".[all]"
) else (
    call .venv\Scripts\activate.bat
)
python -m wmremover
