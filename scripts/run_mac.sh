#!/bin/bash

echo "====================================================="
echo "  Iniciando Mobility Scan (Modo Local para macOS)  "
echo "====================================================="

# Verificar si Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado."
    echo "Por favor, descarga e instala Python 3 desde https://www.python.org/downloads/macos/"
    exit 1
fi

# Moverse a la carpeta raíz del proyecto
cd "$(dirname "$0")/.."

# Crear el entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "📦 Creando entorno virtual aislado de Python..."
    python3 -m venv .venv
fi

# Activar el entorno virtual e instalar/actualizar dependencias
echo "🔄 Instalando/verificando dependencias necesarias..."
source .venv/bin/activate

# Instalamos las dependencias
# Nota: En macOS se requiere opencv-python (no headless) para que aparezca la ventana de la cámara
pip install --upgrade pip -q
pip install mediapipe numpy opencv-python -q

# Crear carpeta de output si no existe
mkdir -p output

# Ejecutar el programa con el modelo optimizado (Lite)
echo "🚀 Iniciando cámara web..."
echo "Presiona la tecla 'ESC' o 'Q' en la ventana de video para salir."
python -m app.main --source 0 --output-dir output --model 0

echo "✅ Ejecución finalizada. Revisa la carpeta 'output/' para ver los resultados."
