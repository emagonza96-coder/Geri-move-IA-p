import cv2
import numpy as np

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
    recording_active: bool = False,
) -> np.ndarray:
    """
    HUD unificado y amigable, totalmente distinto por modo:
      - BODY: panel lateral de ángulos con barras de ROM
      - HAND: panel de 3 métricas grandes + instrucciones para el fisio
    """
    h, w, _ = frame.shape

    # ── BARRA SUPERIOR ────────────────────────────────────────────────────────
    is_hand = active_mode == "HAND"
    is_cardio = active_mode == "CARDIO"

    top_color   = (12, 12, 12)
    
    if is_hand:
        accent = (200, 130, 60)
        mode_label = "MODO MANO  [M] Cambiar Modo"
    elif is_cardio:
        accent = (60, 100, 240)
        mode_label = "MODO CARDIO  [M] Cambiar Modo"
    else:
        accent = (0, 195, 165)
        mode_label = "MODO CUERPO  [M] Cambiar Modo"

    cv2.rectangle(frame, (0, 0), (w, 44), top_color, -1)
    cv2.line(frame, (0, 44), (w, 44), accent, 1, cv2.LINE_AA)

    cv2.putText(frame, "MOBILITY SCAN", (12, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent, 1, cv2.LINE_AA)
    cv2.putText(frame, cached_ts, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)

    # Modo centrado
    (mlw, _), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.putText(frame, mode_label, (w // 2 - mlw // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, accent, 1, cv2.LINE_AA)

    # FPS + mini barra
    fps_color = (80, 230, 100) if fps >= 20 else (60, 165, 255) if fps >= 10 else (60, 80, 255)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 78, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, fps_color, 1, cv2.LINE_AA)
    bx = w - 78
    cv2.rectangle(frame, (bx, 23), (bx + 40, 28), (40, 40, 40), -1)
    fp = int(40 * min(fps / 30.0, 1.0))
    if fp > 0:
        cv2.rectangle(frame, (bx, 23), (bx + fp, 28), fps_color, -1)

    # Indicador REC parpadeante
    if recording_active:
        if int(frame_num * 0.05) % 2 == 0:
            cv2.circle(frame, (w - 18, 16), 6, (40, 40, 240), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (w - 40, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 240), 1, cv2.LINE_AA)

    # ── BARRA INFERIOR ────────────────────────────────────────────────────────
    bar_y = h - 22
    cv2.rectangle(frame, (0, bar_y), (w, h), (10, 10, 10), -1)
    cv2.line(frame, (0, bar_y), (w, bar_y), accent, 1, cv2.LINE_AA)

    if recording_active:
        bottom_text = "[Q] Detener sesion   [P] Pausar   [S] Captura   [H] Panel   [ESC] Salir"
        bottom_color = (100, 100, 240)
    elif is_hand:
        bottom_text = "[Q] Iniciar sesion   [M] Cambiar Modo   [H] Panel   [ESC] Salir"
        bottom_color = (160, 110, 60)
    elif is_cardio:
        bottom_text = "[T] Iniciar 2MST   [M] Cambiar Modo   [H] Panel   [ESC] Salir"
        bottom_color = (100, 130, 240)
    else:
        bottom_text = "[Q] Iniciar sesion   [M] Cambiar Modo   [H] Panel   [R] Reset   [ESC] Salir"
        bottom_color = (80, 160, 140)

    cv2.putText(frame, bottom_text, (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, bottom_color, 1, cv2.LINE_AA)

    if panel_hidden or not angles:
        return frame

    # ── ADVERTENCIAS GLOBALES ────────────────────────────────────────────────
    warnings = set(v.get("warning") for v in angles.values() if isinstance(v, dict) and v.get("warning"))
    if warnings:
        warn_txt = " | ".join(warnings)
        cv2.rectangle(frame, (0, 45), (w, 65), (0, 60, 200), -1)
        cv2.putText(frame, f"AVISO: {warn_txt}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ── PANEL MODO CUERPO (ángulos articulares con ROM) ──────────────────────
    if active_mode == "BODY":
        visible = [(k, v) for k, v in angles.items() if v.get("angle") is not None]
        if not visible:
            return frame

        PANEL_W = 200
        ROW_H   = 30
        panel_h = len(visible) * ROW_H + 30
        px = 8 if panel_side == "left" else w - PANEL_W - 8
        py = 50

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 4, py), (px + PANEL_W + 4, py + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px - 4, py), (px + PANEL_W + 4, py + panel_h), (0, 170, 145), 1)

        cv2.putText(frame, "ANGULOS CORPORALES", (px + 4, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 175), 1, cv2.LINE_AA)
        cv2.line(frame, (px - 4, py + 18), (px + PANEL_W + 4, py + 18), (0, 100, 85), 1)

        y = py + 30
        bar_w = PANEL_W - 16

        for _, data in visible:
            angle_val = data["angle"]
            status    = data.get("status", "dentro_de_referencia")
            rom_range = data.get("rom_range")
            name      = data.get("name", "")

            if status == "dentro_de_referencia":
                color = (80, 225, 110)
            elif status == "fuera_de_referencia":
                color = (60, 150, 255)
            else:
                color = (140, 140, 140)

            cv2.circle(frame, (px + 8, y - 4), 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame, name, (px + 17, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{angle_val:.0f}", (px + PANEL_W - 28, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # Barra ROM
            bar_y2 = y + 5
            cv2.rectangle(frame, (px + 6, bar_y2), (px + 6 + bar_w, bar_y2 + _ROM_BAR_H), (40, 40, 40), -1)
            if rom_range:
                _, max_rom = rom_range
                if max_rom > 0:
                    fp2 = int(bar_w * min(angle_val / max_rom, 1.0))
                    if fp2 > 0:
                        cv2.rectangle(frame, (px + 6, bar_y2), (px + 6 + fp2, bar_y2 + _ROM_BAR_H), color, -1)
                cv2.line(frame, (px + 6 + bar_w - 1, bar_y2),
                         (px + 6 + bar_w - 1, bar_y2 + _ROM_BAR_H), (90, 90, 90), 1)
            y += ROW_H

    # ── PANEL MODO MANO (métricas clínicas resumidas para el fisio) ───────────
    elif active_mode == "HAND":
        # Aquí solo mostramos 3 métricas clave en grande en la esquina IZQUIERDA
        key_metrics = [
            ("Abd. Pulgar", angles.get("abduccion", {}).get("angle"), "gr", (0, 210, 210)),
            ("Cierre Puno", angles.get("cierre",    {}).get("angle"), "%",  (80, 210, 90)),
            ("Kapandji",    angles.get("kapandji",  {}).get("angle"), "/10",(240, 185, 60)),
        ]

        CARD_W, CARD_H = 130, 52
        gap   = 8
        px    = w - 240 + 8
        py    = 52

        for label, val, unit, color in key_metrics:
            # Tarjeta de métrica
            overlay = frame.copy()
            cv2.rectangle(overlay, (px, py), (px + CARD_W, py + CARD_H), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (px, py), (px + CARD_W, py + CARD_H), color, 1)

            # Etiqueta
            cv2.putText(frame, label, (px + 6, py + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)

            # Valor grande
            val_str = f"{val:.0f}{unit}" if val is not None else "--"
            cv2.putText(frame, val_str, (px + 6, py + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            py += CARD_H + gap

        # Instrucción para el fisioterapeuta
        guide_y = py + 6
        if angles.get("kapandji", {}).get("angle", 0) == 0:
            guide = "Acerque el pulgar a cada dedo"
        elif angles.get("cierre", {}).get("angle", 100) < 30:
            guide = "Intente cerrar el puno"
        else:
            guide = "Abra la mano completamente"

        cv2.putText(frame, guide, (px, guide_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 100), 1, cv2.LINE_AA)

    # ── PANEL MODO CARDIO (2MST y Respiración) ───────────
    elif active_mode == "CARDIO":
        CARD_W, CARD_H = 150, 60
        gap = 10
        px = w - 240 + 8
        py = 52
        
        # Tarjeta 1: Frecuencia Respiratoria
        rpm = angles.get("rpm", 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + CARD_W, py + CARD_H), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (px, py), (px + CARD_W, py + CARD_H), (60, 200, 255), 1)
        
        cv2.putText(frame, "FREC. RESPIRATORIA", (px + 6, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
                    
        rpm_str = f"{rpm} RPM" if rpm > 0 else "Calculando..."
        cv2.putText(frame, rpm_str, (px + 6, py + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2, cv2.LINE_AA)
        
        py += CARD_H + gap
        
        # Tarjeta 2: 2-Minute Step Test
        steps = angles.get("steps", 0)
        active = angles.get("step_active", False)
        elapsed = angles.get("step_elapsed", 0.0)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + CARD_W, py + CARD_H), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        
        border_color = (100, 255, 100) if active else (100, 100, 255)
        cv2.rectangle(frame, (px, py), (px + CARD_W, py + CARD_H), border_color, 1)
        
        cv2.putText(frame, "2-MIN STEP TEST", (px + 6, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
                    
        time_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        step_str = f"Pasos: {steps}"
        
        cv2.putText(frame, step_str, (px + 6, py + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)
        cv2.putText(frame, time_str, (px + 6, py + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    
        # Indicadores de estado de rodillas
        state_l = angles.get("state_l", "DOWN")
        state_r = angles.get("state_r", "DOWN")
        
        cl_l = (80, 255, 80) if state_l == "UP" else (100, 100, 100)
        cl_r = (80, 255, 80) if state_r == "UP" else (100, 100, 100)
        
        cv2.circle(frame, (px + CARD_W - 20, py + 35), 6, cl_l, -1)
        cv2.circle(frame, (px + CARD_W - 8, py + 35), 6, cl_r, -1)
        cv2.putText(frame, "L R", (px + CARD_W - 22, py + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    return frame

# CONEXIONES Y NOMBRES PARA DRAW_SKELETON
SKELETON_CONNECTIONS_COLORED = [
    # Tronco (Cian)
    ((11, 12), (200, 200, 40), 2),
    ((11, 23), (200, 200, 40), 2),
    ((12, 24), (200, 200, 40), 2),
    ((23, 24), (200, 200, 40), 2),
    # Brazo Izquierdo (Verde claro)
    ((11, 13), (80, 230, 110), 2),
    ((13, 15), (80, 230, 110), 2),
    # Brazo Derecho (Verde oscuro/Azulado)
    ((12, 14), (60, 150, 100), 2),
    ((14, 16), (60, 150, 100), 2),
    # Pierna Izquierda (Morado/Rosa)
    ((23, 25), (180, 100, 240), 2),
    ((25, 27), (180, 100, 240), 2),
    ((27, 31), (180, 100, 240), 2),
    # Pierna Derecha (Azul claro)
    ((24, 26), (255, 150, 60), 2),
    ((26, 28), (255, 150, 60), 2),
    ((28, 32), (255, 150, 60), 2),
]

LANDMARK_NAMES = {
    0: "nariz",
    11: "hombro_izq", 12: "hombro_der",
    13: "codo_izq", 14: "codo_der",
    15: "muneca_izq", 16: "muneca_der",
    23: "cadera_izq", 24: "cadera_der",
    25: "rodilla_izq", 26: "rodilla_der",
    27: "tobillo_izq", 28: "tobillo_der"
}

def _draw_angles(frame, landmarks, angles, w, h):
    vertex_map = {
        "hombro_izq": 11, "codo_izq": 13, "cadera_izq": 23, "rodilla_izq": 25,
        "hombro_der": 12, "codo_der": 14, "cadera_der": 24, "rodilla_der": 26,
        "muneca_izq": 15, "muneca_der": 16,
    }

    for joint_key, data in angles.items():
        if data["angle"] is None:
            continue
        vertex_idx = vertex_map.get(joint_key)
        if vertex_idx is None:
            continue
        lm = landmarks[vertex_idx]
        if lm.get("visibility", 0) < 0.5:
            continue

        pt = (int(lm["x"] * w), int(lm["y"] * h))
        status = data.get("status", "normal")

        if status == "normal":
            color = (80, 230, 110)
        elif status == "limitado":
            color = (60, 155, 255)
        elif status == "excedido":
            color = (60, 70, 255)
        else:
            color = (200, 200, 200)

        angle_text = f"{data['angle']:.0f}"
        tx, ty = pt[0] + 10, pt[1] - 6

        cv2.rectangle(frame, (tx - 2, ty - 13), (tx + 38, ty + 4), (20, 20, 20), -1)
        cv2.rectangle(frame, (tx - 2, ty - 13), (tx + 38, ty + 4), color, 1)
        cv2.putText(frame, angle_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def draw_skeleton(frame, landmarks, angles=None, panel_hidden=False):
    h, w, _ = frame.shape
    for (idx_a, idx_b), color, thickness in SKELETON_CONNECTIONS_COLORED:
        if idx_a >= len(landmarks) or idx_b >= len(landmarks): continue
        lm_a = landmarks[idx_a]
        lm_b = landmarks[idx_b]
        if lm_a.get("visibility", 0) < 0.5 or lm_b.get("visibility", 0) < 0.5:
            continue
        pt_a = (int(lm_a["x"] * w), int(lm_a["y"] * h))
        pt_b = (int(lm_b["x"] * w), int(lm_b["y"] * h))
        cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    for idx in LANDMARK_NAMES:
        if idx >= len(landmarks): continue
        lm = landmarks[idx]
        if lm.get("visibility", 0) < 0.5:
            continue
        pt = (int(lm["x"] * w), int(lm["y"] * h))
        cv2.circle(frame, pt, 5, (240, 240, 240), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 3, (0, 200, 175), -1, cv2.LINE_AA)

    if angles and not panel_hidden:
        _draw_angles(frame, landmarks, angles, w, h)
    return frame
