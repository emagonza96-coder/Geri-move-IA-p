#!/bin/bash
# ==============================================
# Ejecutar en LINUX con webcam en vivo
# ==============================================
# Permite acceso X11 a Docker y pasa la webcam

set -e

echo "🔧 Permitiendo acceso X11 a Docker..."
xhost +local:docker 2>/dev/null || echo "⚠️  xhost no disponible (¿estás en Wayland?)"

echo "🐳 Ejecutando con webcam en vivo..."
docker compose run --rm \
    --device /dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    app \
    --source 0 \
    --output-dir /data/output

echo "✅ Sesión terminada. Revisa la carpeta output/"
