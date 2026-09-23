@echo off
echo =====================================================
echo   Iniciando Mobility Scan (Modo Local para Windows)
echo =====================================================

REM Moverse a la carpeta raíz del proyecto
cd /d "%~dp0\.."

REM Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no está instalado o no está en el PATH.
    echo Por favor, descarga e instala Python desde https://www.python.org/downloads/windows/
    echo Asegúrate de marcar la opción "Add Python to PATH" durante la instalación.
    pause
    exit /b 1
)

REM Crear el entorno virtual si no existe
if not exist ".venv\" (
    echo [INFO] Creando entorno virtual aislado de Python...
    python -m venv .venv
)

REM Instalar/actualizar dependencias (silenciosamente)
echo [INFO] Instalando dependencias necesarias...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install mediapipe numpy opencv-python -q

REM Crear carpeta de output si no existe
if not exist "output\" mkdir output

echo [INFO] Iniciando cámara web...
echo [INFO] Presiona la tecla 'ESC' o 'Q' en la ventana de video para salir.
python -m app.main --source 0 --output-dir output --model 0

echo =====================================================
echo Ejecución finalizada. Revisa la carpeta 'output\'
echo =====================================================
pause
