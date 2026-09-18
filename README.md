# Detección de Movilidad Corporal — MediaPipe + OpenCV + Docker

Sistema de detección de pose corporal y cálculo de ángulos articulares usando MediaPipe y OpenCV, empaquetado en Docker para funcionar en **Linux, macOS y Windows**.

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) instalado
- Docker Compose (viene incluido con Docker Desktop)

## Estructura del Proyecto

```
├── app/                   # Código fuente
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py            # Script principal
│   └── core/
│       ├── pose_detector.py      # Detector MediaPipe
│       └── angle_calculator.py   # Cálculo de ángulos
├── input/                 # Pon aquí tus videos
├── output/                # Resultados (video + JSON)
├── scripts/               # Scripts de ejecución por plataforma
├── docker-compose.yml
└── README.md
```

## Inicio Rápido

### 1. Construir la imagen

```bash
docker compose build
```

### 2A. Procesar un archivo de video (macOS / Windows / Linux)

```bash
# Coloca tu video en la carpeta input/
cp ~/Videos/mi_video.mp4 input/

# Ejecutar
./scripts/run_video.sh mi_video.mp4

# En Windows:
scripts\run_video.bat mi_video.mp4
```

### 2B. Webcam en vivo (solo Linux)

```bash
./scripts/run_linux_webcam.sh
```

### 3. Ver resultados

Después de ejecutar, revisa la carpeta `output/`:
- `sesion_YYYYMMDD_HHMMSS_*.mp4` — Video con esqueleto y ángulos dibujados
- `sesion_YYYYMMDD_HHMMSS_*_data.json` — Datos de todos los ángulos por frame

## Controles (modo ventana, solo Linux webcam)

| Tecla | Acción |
|-------|--------|
| `Q`   | Salir |
| `S`   | Guardar captura de pantalla |
| `P`   | Pausar / Reanudar |

## Articulaciones Detectadas

| Articulación | Lado Izquierdo | Lado Derecho |
|---|---|---|
| Hombro (flexión) | ✅ | ✅ |
| Codo (flexión)   | ✅ | ✅ |
| Cadera (flexión) | ✅ | ✅ |
| Rodilla (flexión)| ✅ | ✅ |

## Ejemplo de Salida JSON

```json
{
  "session_id": "20260918_091500",
  "total_frames_processed": 450,
  "summary": {
    "rodilla_izq": {
      "name": "Rodilla Izq",
      "min": 95.2,
      "max": 172.8,
      "avg": 145.3,
      "std": 12.1,
      "samples": 430
    }
  }
}
```

## Opciones Avanzadas

```bash
docker compose run --rm app \
    --source /data/input/video.mp4 \
    --headless \
    --confidence 0.8 \
    --model 2 \
    --max-frames 500 \
    --output-dir /data/output
```

| Opción | Descripción | Default |
|---|---|---|
| `--source` | `0` para webcam o ruta a video | `0` |
| `--headless` | Sin ventana (solo procesar) | off |
| `--confidence` | Confianza de detección (0.0-1.0) | 0.7 |
| `--model` | Complejidad: 0=lite, 1=full, 2=heavy | 1 |
| `--max-frames` | Límite de frames (0=sin límite) | 0 |
| `--output-dir` | Directorio de salida | /data/output |

## Compatibilidad por Plataforma

| Función | Linux | macOS | Windows |
|---|---|---|---|
| Procesar video archivo | ✅ | ✅ | ✅ |
| Webcam en vivo | ✅ | ❌ | ❌ |
| Ventana de visualización | ✅ (X11) | ❌ | ❌ |
| Guardar video resultado | ✅ | ✅ | ✅ |
| Guardar datos JSON | ✅ | ✅ | ✅ |

> **Nota:** En macOS y Windows, Docker Desktop no permite acceso directo a la webcam. Graba tu video primero y luego procésalo con el sistema.
