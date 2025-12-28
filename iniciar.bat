@echo off
cd /d "%~dp0"

if not exist "venv" (
    echo ERROR: Ejecuta primero instalar.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python app.py
