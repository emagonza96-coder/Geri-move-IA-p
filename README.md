# Geri-Move IA — Detección de Movilidad Corporal

Sistema avanzado de detección de pose corporal y cálculo de ángulos articulares usando MediaPipe y OpenCV. Diseñado para funcionar en **Linux, macOS y Windows** mediante Docker o entornos virtuales de Python.

## Características Principales

*   **Detección en Tiempo Real:** Usa `PoseLandmarker` de MediaPipe para tracking de alta velocidad.
*   **Lectura sin Lag (ThreadedCapture):** Implementa lectura de video en un hilo separado para evitar el lag de buffer de las webcams.
*   **HUD Optimizado (Heads-Up Display):**
    *   Esqueleto con colores por zona anatómica (torso, brazos, piernas).
    *   Panel de ángulos articulares auto-reubicable con indicador de estado (normal, limitado, excedido).
    *   Barras de progreso visual para el Rango de Movimiento (ROM).
    *   Caché inteligente de renderizado de fuentes para maximizar FPS.
*   **Soporte Multi-Resolución:** Capacidad de negociar resoluciones nativas más rápidas con las webcams (ej: 640x360).
*   **Visualizador 3D:** Herramienta adicional (`view_skeleton_3d.py`) para renderizar el movimiento en un espacio 3D real usando matplotlib.
*   **Preparado para Multi-Cámara:** Incluye módulos base de calibración estéreo y triangulación 3D para futuras expansiones.

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) instalado y Docker Compose, O Python 3.9+ en entorno virtual.

## Estructura del Proyecto

```
├── app/                   # Código fuente Python
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py            # Aplicación principal de 1 cámara
│   ├── multi_main.py      # Implementación base para múltiples cámaras
│   ├── view_skeleton_3d.py# Visualizador 3D de las sesiones
│   └── core/
│       ├── pose_detector.py      # Tracking y dibujo HUD
│       ├── angle_calculator.py   # Cálculos ROM articulares
│       ├── multi_camera.py       # Sincronización multi-cámara
│       ├── calibration.py        # Calibración estéreo (tablero de ajedrez)
│       └── triangulation.py      # Reconstrucción de puntos 3D reales
├── input/                 # Directorio para archivos de video locales
├── output/                # Directorio de resultados (video .mp4 + datos .json)
├── scripts/               # Utilidades de lanzamiento rápido
├── docker-compose.yml
└── README.md
```

## Inicio Rápido (Entorno Virtual Python)

### 1. Preparar Entorno
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
```

### 2. Ejecutar con Webcam Directa
```bash
python app/main.py --source 0 --width 640
```

### 3. Ejecutar con IP Webcam (Celular)
```bash
# Cambia la IP por la que te dé la app de tu teléfono (asegúrate de bajar la resolución en la app a 640x480)
python app/main.py --source "http://192.168.1.169:8080/video" --width 640
```

## Controles en Vivo (Modo Ventana)

| Tecla | Acción |
|-------|--------|
| `Q`   | Salir y guardar sesión |
| `S`   | Guardar captura de pantalla actual |
| `P`   | Pausar / Reanudar stream |
| `H`   | Ocultar / Mostrar panel de ángulos |

## Visualizador 3D

Para ver la representación tridimensional de una sesión guardada:
```bash
python app/view_skeleton_3d.py output/sesion_20260918_130133_webcam_1_data.json
```

## Opciones y Argumentos (main.py)

| Opción | Descripción | Valor por Defecto |
|---|---|---|
| `--source` | ID de cámara (`0`, `1`), URL (`http://...`) o ruta a archivo de video | `0` |
| `--headless` | Modo sin interfaz gráfica (procesamiento más rápido) | off |
| `--confidence` | Nivel mínimo de confianza de detección (0.0 a 1.0) | `0.7` |
| `--model` | Complejidad del modelo (0=lite, 1=full, 2=heavy) | `0` |
| `--max-frames` | Límite máximo de frames a procesar | `0` (infinito) |
| `--width` | Redimensiona / solicita el ancho (ej: `640`) para mejorar los FPS | `0` (nativo) |
| `--panel-side` | Lado de la pantalla para el panel HUD de ángulos (`left`, `right`) | `left` |
| `--output-dir` | Directorio destino para `.mp4` y `.json` | `output/` |

## Docker (Alternativa)

Para correr vía Docker, asegurando que las carpetas de entrada y salida se mapeen correctamente:

### Procesar video de archivo (Todos los OS)
```bash
# Copia un video a la carpeta input/
./scripts/run_video.sh input/mi_video.mp4
```

### Webcam en vivo (Solo Linux)
```bash
./scripts/run_linux_webcam.sh
```

## Ejemplo de Datos Exportados (JSON)

El sistema guarda automáticamente cada sesión registrando ángulos, estados y las coordenadas 3D para análisis posterior.
```json
{
  "session_id": "20260918_132020",
  "fps": 30.0,
  "total_frames_processed": 1889,
  "summary": {
    "codo_der": {
      "name": "Codo Der",
      "min": 0.9,
      "max": 180.0,
      "avg": 119.3
    }
  },
  "measurements": [
    {
      "frame": 1,
      "timestamp": 1789759220.25,
      "angles": {
         "codo_der": {"angle": 135.2, "status": "normal", "visibility": 0.99}
      }
    }
  ]
}
```
