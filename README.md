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
## Requisitos

- [Docker](https://docs.docker.com/get-docker/) instalado y Docker Compose, O Python 3.9+ en entorno virtual.

## Estructura del Proyecto

```
├── app/                   # Código fuente Python
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py            # Aplicación principal
│   ├── view_skeleton_3d.py# Visualizador 3D de las sesiones
│   ├── core/              # Orquestadores y controladores
│   │   ├── session.py     # SessionManager (JSON y VideoWriter)
│   │   └── input.py       # KeyboardController
│   ├── biomechanics/      # Lógica clínica y trigonometría
│   ├── pose/              # Modelos MediaPipe
│   ├── ui/                # Renderizado HUD y gráficos
│   ├── processing/        # Filtros (Savitzky-Golay, etc.)
│   └── quality/           # Evaluador de framing y calidad
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
python -m app.main --source 0
```

### 3. Ejecutar con IP Webcam (Celular)
```bash
# Cambia la IP por la que te dé la app de tu teléfono (asegúrate de bajar la resolución en la app a 640x480)
python -m app.main --source "http://192.168.1.169:8080/video"
```

## Controles en Vivo (Modo Ventana)

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir de la aplicación |
| `Q`   | Iniciar / Detener grabación (solo cuando hay buen encuadre) |
| `O`   | Forzar grabación (ignorar advertencias de encuadre) |
| `P`   | Pausar / Reanudar stream |
| `H`   | Ocultar / Mostrar panel de ángulos |
| `M`   | Cambiar de modo (Cuerpo Completo <-> Mano) |
| `R`   | Resetear la calibración del perfil |
| `S`   | Guardar captura de pantalla actual |

## Visualizador 3D

Para ver la representación tridimensional de una sesión guardada:
```bash
python -m app.view_skeleton_3d output/sesion_20260918_130133_webcam_1_data.json
```

## Opciones y Argumentos (main.py)

| Opción | Descripción | Valor por Defecto |
|---|---|---|
| `--source` | ID de cámara (`0`, `1`), URL (`http://...`) o ruta a archivo de video | `0` |
| `--headless` | Modo sin interfaz gráfica (procesamiento más rápido) | off |
| `--confidence` | Nivel mínimo de confianza de detección (0.0 a 1.0) | `0.7` |
| `--model` | Complejidad del modelo (0=lite, 1=full, 2=heavy) | `1` |
| `--max-frames` | Límite máximo de frames a procesar | `0` (infinito) |
| `--width` | Redimensiona / solicita el ancho (ej: `640`) para mejorar los FPS | `0` (nativo) |
| `--mode` | Modo de análisis inicial (`body` o `hand`) | `body` |
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

## Trabajo Futuro (Roadmap)
Existen áreas de mejora y funcionalidades planeadas que se implementarán en próximas iteraciones:

* **Interfaz CARDIO:** Migrar la interfaz del modo Cardiovascular (Step Test, Frecuencia Respiratoria) al nuevo motor gráfico de renderizado con Pillow.
* **Refactorización de Interacciones:** Trasladar la lógica manual de calibración del ratón (offsets) desde el loop principal hacia un `CalibrationController` dedicado.
* **Sistema Multi-Ventana (PyQt/PySide):** Sustituir el escalado nativo estático de OpenCV por un layout completamente responsivo utilizando Qt, permitiendo al usuario redimensionar la ventana sin perder legibilidad gráfica.
* **Alertas Gráficas Mejoradas:** Migrar los pop-ups integrados sobre el video (como "Paciente no estabilizado") hacia el módulo `VideoOverlay` para mayor cohesión estructural.
