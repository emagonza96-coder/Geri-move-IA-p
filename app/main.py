"""
Sistema de Detección de Movilidad Corporal
==========================================
Usa MediaPipe + OpenCV para detectar pose, calcular ángulos articulares
y guardar los resultados (video procesado + datos JSON).

Modos de uso:
  - Webcam en vivo (Linux con X11):  python main.py --source 0
  - Archivo de video (cualquier SO): python main.py --source /data/input/video.mp4
  - Headless (sin ventana):          python main.py --source /data/input/video.mp4 --headless

Salida:
  - Video procesado con esqueleto y ángulos en /data/output/
  - Datos JSON con todas las mediciones en /data/output/
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.pose_detector import PoseDetector
from core.angle_calculator import calculate_all_angles
from core.hand_detector import HandDetector
from core.hand_metrics import calculate_hand_metrics
from core.calibration_profile import CalibrationProfile


class ThreadedCapture:
    """
    Wrapper de cv2.VideoCapture que lee frames en un hilo separado.
    Siempre entrega el frame más reciente, eliminando el lag del buffer
    que se acumula cuando el procesamiento tarda más que la cámara.
    API compatible con cv2.VideoCapture (read, get, set, isOpened, release).
    """
    def __init__(self, cap: cv2.VideoCapture):
        self._cap     = cap
        self._frame   = None
        self._ret     = False
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def isOpened(self): return self._cap.isOpened()
    def get(self, prop): return self._cap.get(prop)
    def set(self, prop, val): return self._cap.set(prop, val)

    def release(self):
        self._running = False
        self._thread.join(timeout=1.5)
        self._cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detección de movilidad corporal con MediaPipe + OpenCV"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de video: 0 para webcam, o ruta a archivo de video (default: 0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Modo sin ventana (procesa y guarda sin mostrar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directorio de salida para videos y datos (default: /data/output)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confianza mínima de detección (default: 0.7)",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Complejidad del modelo: 0=lite, 1=full, 2=heavy (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Máximo de frames a procesar (0 = sin límite, default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Redimensionar el ancho del frame antes de procesar con MediaPipe (ej: 640). 0 = sin cambio.",
    )
    parser.add_argument(
        "--panel-side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Lado del panel de ángulos: left o right (default: left)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="body",
        choices=["body", "hand"],
        help="Modo de detección: body (cuerpo completo), hand (manos/dedos). Usa tecla M para cambiar en vivo. (default: body)",
    )
    return parser.parse_args()



# Alto de la barra ROM (px). Pre-definido fuera del bucle.
_ROM_BAR_H = 5


def create_hud(
    frame: np.ndarray,
    angles: dict,
    fps: float,
    frame_num: int,
    panel_side: str = "left",
    panel_hidden: bool = False,
    cached_ts: str = "",
    active_mode: str = "BODY",
) -> np.ndarray:
    """
    Dibuja el HUD optimizado:
    - Timestamp cacheado fuera del bucle (sin llamadas al SO por frame)
    - Barra ROM por articulación
    - Indicadores como círculos OpenCV (sin Unicode)
    - Panel posicionable y ocultable con H
    """
    h, w, _ = frame.shape

    # === BARRA SUPERIOR ===
    cv2.rectangle(frame, (0, 0), (w, 44), (10, 10, 10), -1)
    cv2.line(frame, (0, 44), (w, 44), (0, 195, 165), 1, cv2.LINE_AA)

    cv2.putText(frame, "MOBILITY SCAN", (12, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 225, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, cached_ts, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

    # FPS + mini-barra de rendimiento
    fps_color = (80, 240, 100) if fps >= 20 else (60, 165, 255) if fps >= 10 else (60, 80, 255)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 88, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, fps_color, 1, cv2.LINE_AA)

    # Indicador de modo activo (CUERPO / MANO)
    if active_mode == "HAND":
        mode_color = (255, 160, 100)  # Azul-rosa
        mode_label = "MANO"
    else:
        mode_color = (0, 195, 165)
        mode_label = "CUERPO"
    cv2.putText(frame, mode_label, (w // 2 - 30, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)

    bx = w - 88
    cv2.rectangle(frame, (bx, 24), (bx + 44, 30), (40, 40, 40), -1)
    fill_px = int(44 * min(fps / 30.0, 1.0))
    if fill_px > 0:
        cv2.rectangle(frame, (bx, 24), (bx + fill_px, 30), fps_color, -1)
    cv2.putText(frame, f"#{frame_num}", (w - 88, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 100), 1, cv2.LINE_AA)

    # === PANEL DE ÁNGULOS ===
    if angles and not panel_hidden:
        visible = [(k, v) for k, v in angles.items() if v["angle"] is not None]
        if visible:
            PANEL_W = 195
            ROW_H = 33
            panel_h = len(visible) * ROW_H + 28
            panel_x = 8 if panel_side == "left" else w - PANEL_W - 8
            panel_y = 50

            # Fondo sólido oscuro
            cv2.rectangle(frame, (panel_x - 2, panel_y),
                          (panel_x + PANEL_W + 2, panel_y + panel_h), (12, 12, 12), -1)
            cv2.rectangle(frame, (panel_x - 2, panel_y),
                          (panel_x + PANEL_W + 2, panel_y + panel_h), (0, 175, 150), 1, cv2.LINE_AA)

            # Encabezado
            cv2.putText(frame, "ANGULOS", (panel_x + 6, panel_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 185), 1, cv2.LINE_AA)
            cv2.line(frame, (panel_x - 2, panel_y + 20),
                     (panel_x + PANEL_W + 2, panel_y + 20), (0, 140, 120), 1)

            y = panel_y + 32
            bar_w = PANEL_W - 14

            for _, data in visible:
                angle_val = data["angle"]
                status    = data.get("status", "normal")
                rom_range = data.get("rom_range")
                name      = data.get("name", "")

                if status == "normal":
                    color = (80, 230, 110)
                elif status == "limitado":
                    color = (60, 155, 255)
                elif status == "excedido":
                    color = (60, 70, 255)
                else:
                    color = (150, 150, 150)

                # Indicador circular (no Unicode)
                cv2.circle(frame, (panel_x + 8, y - 4), 4, color, -1, cv2.LINE_AA)

                # Nombre articulación
                cv2.putText(frame, name, (panel_x + 18, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 205, 205), 1, cv2.LINE_AA)

                # Ángulo alineado a la derecha
                cv2.putText(frame, f"{angle_val:.0f}", (panel_x + PANEL_W - 30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

                # Barra de progreso ROM
                bar_y = y + 6
                cv2.rectangle(frame, (panel_x + 6, bar_y),
                              (panel_x + 6 + bar_w, bar_y + _ROM_BAR_H), (45, 45, 45), -1)
                if rom_range:
                    _, max_rom = rom_range
                    if max_rom > 0:
                        fill_f = min(angle_val / max_rom, 1.0)
                        fp = int(bar_w * fill_f)
                        if fp > 0:
                            cv2.rectangle(frame, (panel_x + 6, bar_y),
                                          (panel_x + 6 + fp, bar_y + _ROM_BAR_H), color, -1)
                    # Marcador del límite máximo
                    mx = panel_x + 6 + bar_w - 1
                    cv2.line(frame, (mx, bar_y), (mx, bar_y + _ROM_BAR_H), (100, 100, 100), 1)

                y += ROW_H

    # === BARRA INFERIOR ===
    by = h - 24
    cv2.rectangle(frame, (0, by), (w, h), (10, 10, 10), -1)
    cv2.line(frame, (0, by), (w, by), (0, 175, 150), 1, cv2.LINE_AA)
    
    # Textos de la barra inferior según el estado (se actualiza luego en el loop, pero ponemos unos defaults aquí)
    pass

    return frame


def main():
    args = parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determinar fuente de video
    if args.source.isdigit():
        source = int(args.source)
        source_name = f"webcam_{source}"
        print(f"[INFO] Usando webcam: /dev/video{source}")
    else:
        source = args.source
        source_name = Path(source).stem
        is_url = source.startswith(("http://", "https://", "rtsp://"))
        
        if not is_url and not os.path.exists(source):
            print(f"[ERROR] Archivo no encontrado: {source}")
            sys.exit(1)
            
        if is_url:
            print(f"[INFO] Procesando stream de red (celular/IP): {source}")
            source_name = "celular_stream"
        else:
            print(f"[INFO] Procesando video: {source}")

    # Abrir video y configurar resolución antes de empezar a leer
    _raw_cap = cv2.VideoCapture(source)
    if not _raw_cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video: {source}")
        print("[HINT] En macOS/Windows con Docker, usa --source con un archivo de video")
        print("[HINT] En Linux, asegúrate de pasar --device /dev/video0")
        sys.exit(1)

    # Solicitar resolución nativa a la cámara (antes del hilo de lectura)
    if args.width > 0 and isinstance(source, int):
        _raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        _raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
        _raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Propiedades del video (leemos DESPUÉS de solicitar resolución)
    frame_w      = int(_raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(_raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source   = _raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(_raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Activar lector en hilo para eliminar lag del buffer de la cámara
    # (solo para fuentes en tiempo real; archivos de video se leen secuencialmente)
    is_live = isinstance(source, int) or str(source).startswith(("http", "rtsp"))
    if is_live:
        cap = ThreadedCapture(_raw_cap)
        print("[INFO] Modo lector en hilo activo (sin lag de buffer)")
    else:
        cap = _raw_cap


    # Calcular resolución de procesamiento (reducida para mayor FPS)
    if args.width > 0 and args.width < frame_w:
        proc_w = args.width
        proc_h = int(frame_h * (args.width / frame_w))
        print(f"[INFO] Resolución original: {frame_w}x{frame_h} @ {fps_source:.0f} FPS")
        print(f"[INFO] Resolución de procesamiento: {proc_w}x{proc_h} (MediaPipe + HUD)")
    else:
        proc_w, proc_h = frame_w, frame_h
        print(f"[INFO] Resolución: {frame_w}x{frame_h} @ {fps_source:.0f} FPS")

    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")

    # Inicializar detectores según modo inicial (ambos se cargan para permitir cambio con M)
    detector = None
    hand_detector = None

    print(f"[INFO] Inicializando MediaPipe Pose (modelo={args.model}, confianza={args.confidence})")
    detector = PoseDetector(
        min_detection_confidence=args.confidence,
        model_complexity=args.model,
    )

    print(f"[INFO] Inicializando MediaPipe Hands (confianza={args.confidence})")
    hand_detector = HandDetector(
        min_detection_confidence=args.confidence,
    )

    # Preparar escritor de video de salida (siempre en resolución original)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = output_dir / f"sesion_{session_id}_{source_name}.mp4"
    output_data_path = output_dir / f"sesion_{session_id}_{source_name}_data.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps_source,
        (frame_w, frame_h),
    )

    print(f"[INFO] Video de salida: {output_video_path}")
    print(f"[INFO] Datos de salida: {output_data_path}")

    # Variables de sesión
    profile = CalibrationProfile()
    recording_active = False
    
    # Variables para modo ajuste con ratón
    mouse_state = {"dragging": None, "x": 0, "y": 0}
    current_landmarks = [] # Para el callback del ratón

    def on_mouse(event, x, y, flags, param):
        if recording_active or not current_landmarks:
            # Solo permitimos arrastrar en modo calibración (no grabando)
            return
        
        # Coordenadas normalizadas
        nx, ny = x / frame_w, y / frame_h
        mouse_state["x"] = nx
        mouse_state["y"] = ny

        if event == cv2.EVENT_LBUTTONDOWN:
            # Encontrar el landmark más cercano (umbral 5% del ancho de pantalla)
            min_dist = float('inf')
            nearest_idx = None
            for idx, lm in enumerate(current_landmarks):
                if lm.get("visibility", 0) > 0.4:
                    dist = ((lm["x"] - nx)**2 + (lm["y"] - ny)**2)**0.5
                    if dist < 0.05 and dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx
            
            if nearest_idx is not None:
                mouse_state["dragging"] = nearest_idx
                # Para calcular el offset, guardamos la posición original cruda (antes de apply)
                # Como current_landmarks ya tiene offsets aplicados, calculamos la posición original:
                lm_adj = current_landmarks[nearest_idx]
                dx_curr, dy_curr = profile.offsets.get(nearest_idx, (0.0, 0.0))
                mouse_state["orig_x"] = lm_adj["x"] - dx_curr
                mouse_state["orig_y"] = lm_adj["y"] - dy_curr

        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_state["dragging"] is not None:
                idx = mouse_state["dragging"]
                dx = nx - mouse_state["orig_x"]
                dy = ny - mouse_state["orig_y"]
                profile.set_offset(idx, dx, dy)

        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["dragging"] = None

    if not args.headless:
        print(f"[INFO] Modo visual activo — presiona Q para salir")
        cv2.namedWindow("Mobility Scan - MediaPipe + OpenCV")
        cv2.setMouseCallback("Mobility Scan - MediaPipe + OpenCV", on_mouse)
    else:
        print(f"[INFO] Modo headless — procesando sin ventana...")

    # Variables de estado
    all_measurements = []
    frame_count      = 0
    recorded_frames  = 0
    paused           = False
    panel_hidden     = False
    prev_time        = time.time()
    fps_display      = 0.0
    active_mode      = "HAND" if args.mode == "hand" else "BODY"
    cached_ts        = datetime.now().strftime("%H:%M:%S")
    ts_update_time   = time.time()
    screenshot_time  = 0.0

    try:
        while cap.isOpened():
            key = cv2.waitKey(1) & 0xFF
            
            # Procesar teclas globales primero
            if key == 27: # ESC
                print("[INFO] Salida solicitada por usuario")
                break
            elif key == ord("q"):
                recording_active = not recording_active
                if recording_active:
                    print("[INFO] GRABANDO SECCIÓN")
                else:
                    print("[INFO] GRABACIÓN DETENIDA (Modo Calibración)")

            if paused:
                if key == ord("p"):
                    paused = False
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # ThreadedCapture aún no tiene su primer frame (inicio) o
                # cámara física perdió un frame — simplemente saltamos sin log
                time.sleep(0.005)
                continue
            if not isinstance(cap, ThreadedCapture) and not ret:
                # Solo para archivos de video: fin del archivo
                print("[INFO] Fin del video")
                break

            frame_count += 1

            # Límite de frames
            if args.max_frames > 0 and frame_count > args.max_frames:
                print(f"[INFO] Límite de {args.max_frames} frames alcanzado")
                break

            # Calcular FPS real
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps_display = 0.8 * fps_display + 0.2 * (1.0 / dt)
            prev_time = current_time

            # Reducir resolución para MediaPipe si se solicitó (mayor FPS)
            if proc_w != frame_w:
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                proc_frame = frame

            # Detectar según el modo activo
            angles = {}
            detected = False
            raw_landmarks = []

            if active_mode == "BODY" and detector:
                raw_landmarks = detector.detect(proc_frame)
                if raw_landmarks:
                    detected = True
                    # Aplicar calibración
                    current_landmarks = profile.apply(raw_landmarks)
                    angles = calculate_all_angles(current_landmarks)

                    if recording_active:
                        all_measurements.append({
                            "frame": recorded_frames,
                            "timestamp": current_time,
                            "mode": "body",
                            "angles": {
                                jk: {"angle": d["angle"], "status": d["status"], "visibility": d["visibility"]}
                                for jk, d in angles.items()
                            },
                        })
                    frame = detector.draw_skeleton(frame, current_landmarks, angles)

            elif active_mode == "HAND" and hand_detector:
                hands = hand_detector.detect(proc_frame)
                if hands:
                    detected = True
                    for i, hand_data in enumerate(hands):
                        # Aplicar calibración (usamos offset general para landmarks 0-20, asumiendo 1 mano para simplicidad de ajuste, aunque lo ideal sería offsets por mano)
                        # Por ahora aplicamos offsets a los landmarks de la mano.
                        calibrated_hand_lms = profile.apply(hand_data["landmarks"])
                        hand_data["landmarks"] = calibrated_hand_lms
                        if i == 0:
                            current_landmarks = calibrated_hand_lms # Para el ratón, solo la primera mano

                        hand_angles = calculate_hand_metrics(
                            calibrated_hand_lms,
                            handedness=hand_data["handedness"],
                        )
                        angles.update(hand_angles)

                    if recording_active:
                        all_measurements.append({
                            "frame": recorded_frames,
                            "timestamp": current_time,
                            "mode": "hand",
                            "angles": {
                                jk: {"angle": d["angle"], "status": d["status"], "visibility": d["visibility"]}
                                for jk, d in angles.items()
                            },
                        })
                    frame = hand_detector.draw_hand(frame, hands, angles)
            
            if not detected:
                current_landmarks = []

            if not detected:
                cx, cy = frame_w // 2, frame_h // 2
                cv2.rectangle(frame, (cx - 115, cy - 20), (cx + 115, cy + 10), (20, 20, 20), -1)
                cv2.rectangle(frame, (cx - 115, cy - 20), (cx + 115, cy + 10), (0, 60, 200), 1)
                cv2.putText(frame, "SIN DETECCION", (cx - 110, cy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 100, 255), 2, cv2.LINE_AA)

            # Actualizar timestamp cacheado (1 vez por segundo)
            now = time.time()
            if now - ts_update_time >= 1.0:
                cached_ts = datetime.now().strftime("%H:%M:%S")
                ts_update_time = now

            # Dibujar HUD
            frame = create_hud(
                frame, angles, fps_display, frame_count,
                panel_side=args.panel_side,
                panel_hidden=panel_hidden,
                cached_ts=cached_ts,
                active_mode=active_mode,
            )

            # Escribir textos en la barra inferior y HUD de estado
            h_f, w_f = frame.shape[:2]
            if recording_active:
                cv2.putText(frame, "GRABANDO SESION... [Q] Detener  [P] Pausa  [ESC] Salir  [H] Panel  [M] Modo",
                            (10, h_f - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Indicador REC debajo de la barra superior (derecha) para no sobreponerse
                if time.time() % 1.0 > 0.4:
                    cv2.circle(frame, (w_f - 100, 65), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (w_f - 85, 71),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Borde rojo alrededor del video para indicar grabación activa
                cv2.rectangle(frame, (0, 0), (w_f-1, h_f-1), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "CALIBRACION: Arrastre los puntos. [Q] Iniciar Grabacion  [ESC] Salir  [R] Reset",
                            (10, h_f - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Indicador claro debajo de la barra superior (centro)
                cv2.putText(frame, "MODO CALIBRACION - NO GRABANDO", (w_f // 2 - 160, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Confirmación visual de screenshot
            dt_screen = time.time() - screenshot_time
            if dt_screen < 1.0:
                # Texto flotante
                cv2.putText(frame, "CAPTURA GUARDADA", (w_f // 2 - 110, h_f // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Resaltar landmarks ajustados solo en modo calibración
            if not recording_active:
                for idx, lm in enumerate(current_landmarks):
                    if lm.get("_adjusted", False):
                        px, py = int(lm["x"] * frame_w), int(lm["y"] * frame_h)
                        cv2.circle(frame, (px, py), 6, (0, 215, 255), -1) # Dorado
                        cv2.circle(frame, (px, py), 8, (0, 100, 200), 1)

            # Escribir frame al video de salida solo si estamos grabando
            if recording_active:
                writer.write(frame)
                recorded_frames += 1

            # Mostrar ventana si no es headless
            if not args.headless:
                cv2.imshow("Mobility Scan - MediaPipe + OpenCV", frame)

            # Teclas de acción locales procesadas aquí (las de flujo ya se checan arriba en cv2.waitKey(1))
            if key != 255:
                if key == ord("s"):
                    screenshot_path = output_dir / f"captura_{session_id}_f{frame_count}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                    screenshot_time = time.time()
                    print(f"[INFO] Captura guardada: {screenshot_path}")
                elif key == ord("p"):
                    paused = True
                    print("[INFO] Pausado — presiona P para continuar")
                elif key == ord("h"):
                    panel_hidden = not panel_hidden
                    estado = "oculto" if panel_hidden else "visible"
                    print(f"[INFO] Panel de ángulos {estado}")
                elif key == ord("m"):
                    # Cambio manual de modo
                    if active_mode == "BODY" and hand_detector:
                        active_mode = "HAND"
                    elif active_mode == "HAND" and detector:
                        active_mode = "BODY"
                    print(f"[INFO] Modo cambiado manualmente a: {active_mode}")
                elif key == ord("r") and not recording_active:
                    profile.reset()
                    print(f"[INFO] Calibración reseteada")

            # Progreso para archivos de video
            if total_frames > 0 and frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"[INFO] Progreso: {progress:.1f}% ({frame_count}/{total_frames})")

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario")

    finally:
        # Guardar datos JSON
        session_data = {
            "session_id": session_id,
            "source": str(source),
            "resolution": f"{frame_w}x{frame_h}",
            "fps": fps_source,
            "total_frames_processed": recorded_frames,
            "model_complexity": args.model,
            "detection_mode": args.mode,
            "detection_confidence": args.confidence,
            "measurements": all_measurements,
            "summary": generate_summary(all_measurements),
        }

        with open(output_data_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)

        # Liberar recursos
        cap.release()
        writer.release()
        if detector:
            detector.release()
        if hand_detector:
            hand_detector.release()
        if not args.headless:
            cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"  SESIÓN COMPLETADA")
        print(f"{'='*50}")
        print(f"  Frames visualizados: {frame_count}")
        print(f"  Frames grabados:     {recorded_frames}")
        print(f"  Video guardado:      {output_video_path}")
        print(f"  Datos guardados:     {output_data_path}")
        print(f"{'='*50}")

        # Mostrar resumen de ángulos
        if all_measurements:
            summary = session_data["summary"]
            print(f"\n  RESUMEN DE ÁNGULOS (promedios)")
            print(f"  {'─'*40}")
            for joint_key, stats in summary.items():
                print(
                    f"  {stats['name']:16s}  "
                    f"min={stats['min']:5.1f}°  "
                    f"max={stats['max']:5.1f}°  "
                    f"avg={stats['avg']:5.1f}°"
                )
            print()


def generate_summary(measurements: List[dict]) -> dict:
    """Genera un resumen estadístico de todas las mediciones de la sesión."""
    if not measurements:
        return {}

    # Agrupar valores de ángulo por articulación en un solo recorrido
    angle_data: dict = {}
    for m in measurements:
        for joint_key, data in m["angles"].items():
            if data["angle"] is not None:
                if joint_key not in angle_data:
                    angle_data[joint_key] = []
                angle_data[joint_key].append(data["angle"])

    # Calcular estadísticas — el nombre canónico viene de JOINT_ANGLES, no del JSON
    from core.angle_calculator import JOINT_ANGLES
    summary = {}
    for joint_key, values in angle_data.items():
        if values:
            summary[joint_key] = {
                "name": JOINT_ANGLES.get(joint_key, {}).get("name", joint_key),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "avg": round(sum(values) / len(values), 1),
                "std": round(float(np.std(values)), 1),
                "samples": len(values),
            }

    return summary


if __name__ == "__main__":
    main()
