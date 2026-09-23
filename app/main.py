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

from app.pose.mediapipe_impl import MediaPipePoseEstimator
from app.biomechanics.angles import calculate_all_angles
from app.pose.hand import HandDetector
from app.biomechanics.hand import evaluate_biomechanics
from app.processing.profile import CalibrationProfile
from app.ui.overlay import draw_skeleton


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
    hand_metrics: dict = None,
) -> np.ndarray:
    """
    Dibuja el sidebar completo (panel derecho de 240px).
    La cámara (área izquierda) NO es tocada por esta función.
    - Modo CUERPO: muestra barras de ROM por articulación.
    - Modo MANO: muestra goniometría por dedo + Kapandji + Cierre.
    """
    h, w, _ = frame.shape
    SIDEBAR_W = 240
    px0 = w - SIDEBAR_W  # Inicio del sidebar (ej. 640 si frame=880)
    PAD = 10

    # ── Fondo del sidebar ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (px0, 0), (w, h), (14, 14, 18), -1)

    # ── Separador vertical (línea teal) ───────────────────────────────────────
    cv2.line(frame, (px0, 0), (px0, h), (0, 195, 165), 1, cv2.LINE_AA)

    # ── ENCABEZADO ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (px0, 0), (w, 46), (20, 20, 26), -1)
    cv2.line(frame, (px0, 46), (w, 46), (0, 185, 155), 1, cv2.LINE_AA)

    cv2.putText(frame, "MOBILITY SCAN", (px0 + PAD, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, cached_ts, (px0 + PAD, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 115), 1, cv2.LINE_AA)

    # FPS con mini-barra de carga
    fps_color = (80, 240, 100) if fps >= 20 else (60, 165, 255) if fps >= 10 else (60, 80, 255)
    cv2.putText(frame, f"{fps:.0f}fps", (w - 52, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, fps_color, 1, cv2.LINE_AA)
    bar_bx = w - 52
    cv2.rectangle(frame, (bar_bx, 24), (bar_bx + 40, 30), (40, 40, 40), -1)
    fp = int(40 * min(fps / 30.0, 1.0))
    if fp > 0:
        cv2.rectangle(frame, (bar_bx, 24), (bar_bx + fp, 30), fps_color, -1)

    # Badge de modo (pill indicator)
    if active_mode == "HAND":
        mode_color = (255, 140, 80)
        mode_label = "MANO"
    else:
        mode_color = (0, 195, 165)
        mode_label = "CUERPO"
    cv2.rectangle(frame, (w - 60, 32), (w - 4, 44), (30, 30, 36), -1)
    cv2.putText(frame, mode_label, (w - 57, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, mode_color, 1, cv2.LINE_AA)

    cy = 54  # cursor y inicial, debajo del header

    # ── PANEL MODO CUERPO: ROM por articulación ───────────────────────────────
    if active_mode == "BODY" and angles and not panel_hidden:
        visible = [(k, v) for k, v in angles.items() if v.get("angle") is not None]
        if visible:
            PANEL_W = SIDEBAR_W - PAD * 2
            ROW_H = 32

            # Título de sección
            cv2.putText(frame, "ANGULOS ROM", (px0 + PAD, cy + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 175), 1, cv2.LINE_AA)
            cv2.line(frame, (px0 + PAD, cy + 17), (w - PAD, cy + 17), (0, 120, 105), 1)
            cy += 22

            bar_w = PANEL_W - 50

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

                # Indicador de estado + nombre articulación
                cv2.circle(frame, (px0 + PAD + 4, cy + 7), 4, color, -1, cv2.LINE_AA)
                cv2.putText(frame, name, (px0 + PAD + 14, cy + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 210), 1, cv2.LINE_AA)
                # Ángulo numérico (alineado a la derecha)
                cv2.putText(frame, f"{angle_val:.0f}", (w - 34, cy + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # Barra ROM
                bar_y = cy + 16
                cv2.rectangle(frame, (px0 + PAD, bar_y), (px0 + PAD + bar_w, bar_y + 5), (40, 40, 46), -1)
                if rom_range:
                    _, max_rom = rom_range
                    if max_rom > 0:
                        fill_f = min(angle_val / max_rom, 1.0)
                        fp2 = int(bar_w * fill_f)
                        if fp2 > 0:
                            cv2.rectangle(frame, (px0 + PAD, bar_y),
                                          (px0 + PAD + fp2, bar_y + 5), color, -1)

                cy += ROW_H

    # ── PANEL MODO MANO: Goniometría clínica ──────────────────────────────────
    elif active_mode == "HAND" and hand_metrics:
        gonio = hand_metrics.get("goniometria", {})
        abd   = hand_metrics.get("abduccion_pulgar")
        cierre = hand_metrics.get("indice_cierre", 0.0)
        kapandji = hand_metrics.get("kapandji_nivel", 0)
        deformities = hand_metrics.get("deformidades", [])

        # Título
        cv2.putText(frame, "GONIOMETRIA MANO", (px0 + PAD, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 160, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (px0 + PAD, cy + 17), (w - PAD, cy + 17), (120, 75, 35), 1)
        cy += 22

        finger_rows = [
            ("PULGAR",  [("MCF", gonio.get("pulgar_mcf")), ("IF",  gonio.get("pulgar_if"))]),
            ("INDICE",  [("MCF", gonio.get("indice_mcf")), ("IFP", gonio.get("indice_ifp")), ("IFD", gonio.get("indice_ifd"))]),
            ("MEDIO",   [("MCF", gonio.get("medio_mcf")),  ("IFP", gonio.get("medio_ifp")),  ("IFD", gonio.get("medio_ifd"))]),
            ("ANULAR",  [("MCF", gonio.get("anular_mcf")), ("IFP", gonio.get("anular_ifp")), ("IFD", gonio.get("anular_ifd"))]),
            ("MENIQUE", [("MCF", gonio.get("menique_mcf")),("IFP", gonio.get("menique_ifp")),("IFD", gonio.get("menique_ifd"))]),
        ]

        ROW_H    = 13
        HEADER_H = 15

        for finger_name, joints in finger_rows:
            cv2.putText(frame, finger_name, (px0 + PAD, cy + HEADER_H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 185), 1, cv2.LINE_AA)
            cy += HEADER_H

            for j_label, j_val in joints:
                if j_val is None:
                    angle_str, bar_fill, color = "--", 0, (70, 70, 80)
                else:
                    angle_str = f"{j_val:.0f}"
                    bar_fill = int(max(0, min(j_val, 120)) / 120 * 70)
                    if j_val < -5 or j_val > 110:
                        color = (60, 60, 240)
                    elif "IFP" in j_label:
                        color = (80, 215, 255)
                    elif "MCF" in j_label:
                        color = (150, 150, 40)
                    else:
                        color = (90, 185, 100)

                cv2.putText(frame, j_label, (px0 + PAD, cy + ROW_H - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (110, 110, 125), 1, cv2.LINE_AA)
                bx_bar = px0 + PAD + 26
                bx_end = bx_bar + 70
                bar_y  = cy + ROW_H - 7
                cv2.rectangle(frame, (bx_bar, bar_y), (bx_end, bar_y + 4), (35, 35, 42), -1)
                if bar_fill > 0:
                    cv2.rectangle(frame, (bx_bar, bar_y), (bx_bar + bar_fill, bar_y + 4), color, -1)
                cv2.putText(frame, angle_str, (bx_end + 4, cy + ROW_H - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.29, color, 1, cv2.LINE_AA)
                cy += ROW_H

            cv2.line(frame, (px0 + PAD, cy + 1), (w - PAD, cy + 1), (30, 30, 38), 1)
            cy += 4

        # Resumen global
        cy += 4
        cv2.line(frame, (px0 + PAD, cy), (w - PAD, cy), (0, 140, 120), 1)
        cy += 8
        cv2.putText(frame, "RESUMEN", (px0 + PAD, cy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 160, 80), 1, cv2.LINE_AA)
        cy += 15

        abd_str = f"{abd:.0f}deg" if abd is not None else "--"
        cv2.putText(frame, f"Abd.Pulgar  {abd_str}", (px0 + PAD, cy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 220, 220), 1, cv2.LINE_AA)
        cy += 14

        pct = int((cierre or 0) * 100)
        c_color = (60, 220, 80) if pct > 70 else (60, 130, 255)
        cv2.putText(frame, "Cierre puno", (px0 + PAD, cy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 175), 1, cv2.LINE_AA)
        bpx = px0 + PAD + 78
        cv2.rectangle(frame, (bpx, cy + 2), (bpx + 50, cy + 9), (35, 35, 42), -1)
        cv2.rectangle(frame, (bpx, cy + 2), (bpx + int(50 * pct / 100), cy + 9), c_color, -1)
        cv2.putText(frame, f"{pct}%", (bpx + 54, cy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, c_color, 1, cv2.LINE_AA)
        cy += 14

        kap_color = (60, 220, 80) if kapandji >= 8 else (80, 200, 200) if kapandji > 4 else (60, 130, 255)
        cv2.putText(frame, f"Kapandji    {kapandji}/10", (px0 + PAD, cy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, kap_color, 1, cv2.LINE_AA)
        cy += 14

        if deformities:
            for d in deformities[:3]:
                cv2.putText(frame, f"! {d[:25]}", (px0 + PAD, cy + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (60, 60, 240), 1, cv2.LINE_AA)
                cy += 13

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
    detector = MediaPipePoseEstimator(
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
        (frame_w + 240, frame_h),
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
                
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if "last_timestamp_ms" not in locals():
                last_timestamp_ms = -1
            if timestamp_ms <= 0 or timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 33  # ~30 FPS
            last_timestamp_ms = timestamp_ms

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
                raw_landmarks = detector.detect(proc_frame, timestamp_ms)
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
                    frame = draw_skeleton(frame, current_landmarks, angles, panel_hidden)

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

                        hand_angles = evaluate_biomechanics(calibrated_hand_lms)
                        if i == 0:
                            angles = hand_angles  # Metricas de la primera mano para el sidebar

                    if recording_active:
                        all_measurements.append({
                            "frame": recorded_frames,
                            "timestamp": current_time,
                            "mode": "hand",
                        })
                    deformities = angles.get("deformidades", []) if isinstance(angles, dict) else []
                    frame = hand_detector.draw_hand(frame, hands,
                                                   cam_w=frame_w,
                                                   deformities=deformities)
            
            if not detected:
                current_landmarks = []

            if not detected:
                px0 = frame_w
                cv2.rectangle(frame, (px0 + 20, 65), (px0 + 220, 95), (20, 20, 20), -1)
                cv2.rectangle(frame, (px0 + 20, 65), (px0 + 220, 95), (0, 60, 200), 1)
                cv2.putText(frame, "SIN DETECCION", (px0 + 40, 87),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 100, 255), 2, cv2.LINE_AA)

            # Actualizar timestamp cacheado (1 vez por segundo)
            now = time.time()
            if now - ts_update_time >= 1.0:
                cached_ts = datetime.now().strftime("%H:%M:%S")
                ts_update_time = now

            # Expandir el frame para que el HUD quede en un panel lateral negro
            SIDEBAR_W = 240
            canvas = np.zeros((frame_h, frame_w + SIDEBAR_W, 3), dtype=np.uint8)
            canvas[:, :frame_w] = frame
            frame = canvas

            # Dibujar sidebar unificado
            hud_angles = angles if active_mode == "BODY" else {}
            hud_hand   = angles if active_mode == "HAND" else None
            frame = create_hud(
                frame, hud_angles, fps_display, frame_count,
                panel_hidden=False,
                cached_ts=cached_ts,
                active_mode=active_mode,
                hand_metrics=hud_hand,
            )

            # Escribir textos en la barra inferior y HUD de estado (aislado al panel lateral)
            h_f, w_f = frame.shape[:2]
            px0 = w_f - SIDEBAR_W
            
            if recording_active:
                # Fondo oscuro para la zona inferior del panel
                cv2.rectangle(frame, (px0, h_f - 60), (w_f, h_f), (15, 15, 15), -1)
                
                cv2.putText(frame, "GRABANDO SESION", (px0 + 10, h_f - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "[Q] Detener [P] Pausa", (px0 + 10, h_f - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(frame, "[ESC] Salir [M] Modo", (px0 + 10, h_f - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
                
                # Indicador REC debajo de la barra superior (en el panel lateral)
                if time.time() % 1.0 > 0.4:
                    cv2.circle(frame, (px0 + 30, 65), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (px0 + 45, 71),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Borde rojo alrededor DEL PANEL LATERAL para indicar grabación activa (sin tocar la cámara)
                cv2.rectangle(frame, (px0, 0), (w_f-1, h_f-1), (0, 0, 255), 2)
            else:
                # Fondo oscuro para la zona inferior del panel
                cv2.rectangle(frame, (px0, h_f - 75), (w_f, h_f), (15, 15, 15), -1)
                
                cv2.putText(frame, "MODO CALIBRACION", (px0 + 10, h_f - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "NO GRABANDO", (px0 + 10, h_f - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                            
                cv2.putText(frame, "[Q] Iniciar [R] Reset", (px0 + 10, h_f - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(frame, "[ESC] Salir [M] Modo", (px0 + 10, h_f - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

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
        for joint_key, data in m.get("angles", {}).items():
            if data.get("angle") is not None:
                if joint_key not in angle_data:
                    angle_data[joint_key] = []
                angle_data[joint_key].append(data["angle"])

    # Calcular estadísticas — el nombre canónico viene de JOINT_ANGLES, no del JSON
    from app.biomechanics.angles import JOINT_ANGLES
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
