@echo off
echo ========================================
echo Whisper Transcriber - Instalacion
echo ========================================
echo.

cd /d "%~dp0"

echo Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no encontrado
    echo Descargar de: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Creando entorno virtual...
if not exist "venv" (
    python -m venv venv
)

echo.
echo Activando entorno...
call venv\Scripts\activate.bat

echo.
echo Instalando dependencias (esto puede tardar varios minutos)...
echo.

pip install --upgrade pip

echo [1/4] Instalando Faster-Whisper...
pip install faster-whisper

echo [2/4] Instalando webrtcvad (version precompilada)...
pip install webrtcvad-wheels

echo [3/4] Instalando Resemblyzer para deteccion de hablantes...
pip install --no-deps resemblyzer
pip install torch scipy librosa

echo [4/4] Instalando dependencias adicionales...
pip install numpy

echo.
echo ========================================
echo Instalacion completada!
echo ========================================
echo.
echo Para iniciar la aplicacion ejecuta: iniciar.bat
echo.
pause
