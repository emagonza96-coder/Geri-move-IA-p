@echo off
REM ==============================================
REM Ejecutar en Windows - Procesa archivo de video
REM ==============================================
REM Uso: run_video.bat mi_video.mp4

if "%~1"=="" (
    echo ❌ Uso: run_video.bat ^<nombre_del_video.mp4^>
    echo 📂 Coloca tu video en la carpeta input\ y vuelve a ejecutar
    exit /b 1
)

set VIDEO_FILE=%~1

if not exist "input\%VIDEO_FILE%" (
    echo ❌ No se encontro: input\%VIDEO_FILE%
    echo 📂 Coloca tu video en la carpeta input\ y vuelve a ejecutar
    exit /b 1
)

echo 🐳 Procesando video: %VIDEO_FILE%
echo ⏳ Esto puede tomar un momento...

docker compose run --rm app --source "/data/input/%VIDEO_FILE%" --headless --output-dir /data/output

echo.
echo ✅ Listo! Revisa la carpeta output\ para ver:
echo    📹 Video procesado con esqueleto y angulos
echo    📊 Archivo JSON con todas las mediciones
