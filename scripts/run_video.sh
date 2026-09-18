#!/bin/bash
# ==============================================
# Ejecutar en macOS / Windows / Linux
# Procesa un archivo de video (modo headless)
# ==============================================
# Uso: ./run_video.sh mi_video.mp4

set -e

VIDEO_FILE="${1:?❌ Uso: ./run_video.sh <nombre_del_video.mp4>}"

if [ ! -f "input/$VIDEO_FILE" ]; then
    echo "❌ No se encontró: input/$VIDEO_FILE"
    echo "📂 Coloca tu video en la carpeta input/ y vuelve a ejecutar"
    exit 1
fi

echo "🐳 Procesando video: $VIDEO_FILE"
echo "⏳ Esto puede tomar un momento..."

docker compose run --rm \
    app \
    --source "/data/input/$VIDEO_FILE" \
    --headless \
    --output-dir /data/output

echo ""
echo "✅ ¡Listo! Revisa la carpeta output/ para ver:"
echo "   📹 Video procesado con esqueleto y ángulos"
echo "   📊 Archivo JSON con todas las mediciones"
