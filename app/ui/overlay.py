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
    hand_metrics: dict = None,
    session_info: dict = None,
    quality_indicator: float = 100.0,
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
    HEADER_H = 75
    cv2.rectangle(frame, (px0, 0), (w, HEADER_H), (20, 20, 26), -1)
    cv2.line(frame, (px0, HEADER_H), (w, HEADER_H), (0, 185, 155), 1, cv2.LINE_AA)

    # Título y Reloj
    cv2.putText(frame, "MOBILITY SCAN", (px0 + PAD, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 190), 1, cv2.LINE_AA)
    
    cv2.putText(frame, cached_ts, (px0 + PAD, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 115), 1, cv2.LINE_AA)

    # Info de Sesión (Task 6)
    if not session_info:
        session_info = {"id": "Anon-1", "task": "Libre", "intent": 1, "view": "Frontal", "timer": "00:00"}
    
    sid = session_info.get('id', 'N/A')
    task = session_info.get('task', 'N/A')
    intent = session_info.get('intent', 1)
    view = session_info.get('view', 'Frontal')
    timer = session_info.get('timer', '00:00')
    
    cv2.putText(frame, f"ID: {sid} | Tarea: {task} (#{intent})", (px0 + PAD, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vista: {view} | Timer: {timer}", (px0 + PAD, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS
    fps_color = (80, 240, 100) if fps >= 20 else (60, 165, 255) if fps >= 10 else (60, 80, 255)
    cv2.putText(frame, f"{fps:.0f}fps", (w - 45, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, fps_color, 1, cv2.LINE_AA)
                
    # Indicador de Calidad (Task 7)
    qual_color = (80, 240, 100) if quality_indicator >= 85 else (0, 140, 255) if quality_indicator >= 50 else (60, 60, 255)
    cv2.putText(frame, f"Q: {quality_indicator:.0f}%", (w - 55, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, qual_color, 1, cv2.LINE_AA)

    # Badge de modo (pill indicator)
    if active_mode == "HAND":
        mode_color = (255, 140, 80)
        mode_label = "MANO"
    else:
        mode_color = (0, 195, 165)
        mode_label = "CUERPO"
    cv2.rectangle(frame, (w - 55, 42), (w - 4, 54), (30, 30, 36), -1)
    cv2.putText(frame, mode_label, (w - 52, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, mode_color, 1, cv2.LINE_AA)

    cy = HEADER_H + 8  # cursor y inicial, debajo del header

    # ── PANEL MODO CUERPO: ROM por articulación ───────────────────────────────
    if active_mode == "BODY" and angles and not panel_hidden:
        PANEL_W = SIDEBAR_W - PAD * 2
        
        cv2.putText(frame, "ANGULOS ROM", (px0 + PAD, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 175), 1, cv2.LINE_AA)
        cv2.line(frame, (px0 + PAD, cy + 17), (w - PAD, cy + 17), (0, 120, 105), 1)
        cy += 25

        LAYOUT_ORDEN = [
            ("cuello_inclinacion", "cuello_rotacion"),
            ("tronco_inclinacion", "tronco_rotacion"),
            ("hombro_izq", "hombro_der"),
            ("codo_izq", "codo_der"),
            ("cadera_izq", "cadera_der"),
            ("rodilla_izq", "rodilla_der"),
            ("muneca_izq", "muneca_der"),
        ]

        COL_W = PANEL_W // 2
        ROW_H = 34
        
        for keys_row in LAYOUT_ORDEN:
            for c_idx, key in enumerate(keys_row):
                if not key: continue
                data = angles.get(key)
                if not data: continue
                
                cx = px0 + PAD + (c_idx * COL_W)
                
                name = data.get("name", "").replace(" Izq", "").replace(" Der", "")
                name = f"{name} (I)" if "izq" in key else f"{name} (D)" if "der" in key else name
                
                angle_val = data.get("angle")
                status = data.get("status", "fuera_de_cuadro")
                rom_range = data.get("rom_range")
                
                if status == "normal" or status == "dentro_de_referencia":
                    color = (80, 230, 110)
                elif status == "limitado" or status == "fuera_de_referencia":
                    color = (60, 155, 255)
                elif status == "excedido":
                    color = (60, 70, 255)
                else: # fuera_de_cuadro, no_visible
                    color = (110, 110, 110)

                # Nombre articulacion
                cv2.circle(frame, (cx + 4, cy + 4), 3, color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[:15], (cx + 12, cy + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 210), 1, cv2.LINE_AA)
                
                # Angulo
                if angle_val is not None:
                    txt = f"{angle_val:.0f}"
                else:
                    txt = "--"
                
                # Aliñado a la derecha de su columna
                cv2.putText(frame, txt, (cx + COL_W - 24, cy + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                
                # Barra de rango de referencia (Task 5)
                bar_y = cy + 14
                bar_w = COL_W - 12
                cv2.rectangle(frame, (cx, bar_y), (cx + bar_w, bar_y + 4), (40, 40, 46), -1)
                
                if angle_val is not None and rom_range:
                    r_min, r_max = rom_range
                    if r_max > 0:
                        scale_max = max(r_max * 1.2, 1)
                        # Dibujar rango esperado (gris más claro en el fondo)
                        x_min = int((r_min / scale_max) * bar_w)
                        x_max = int((r_max / scale_max) * bar_w)
                        cv2.rectangle(frame, (cx + x_min, bar_y), (cx + x_max, bar_y + 4), (70, 70, 75), -1)
                        
                        # Dibujar valor real
                        fill_f = min(angle_val / scale_max, 1.0)
                        fp2 = int(bar_w * fill_f)
                        if fp2 > 0:
                            cv2.rectangle(frame, (cx, bar_y), (cx + fp2, bar_y + 4), color, -1)

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
    
    drawn_boxes = []

    for joint_key, data in angles.items():
        if data.get("angle") is None:
            continue
        vertex_idx = vertex_map.get(joint_key)
        if vertex_idx is None:
            continue
        lm = landmarks[vertex_idx]
        if lm.get("visibility", 0) < 0.5:
            continue

        pt = (int(lm["x"] * w), int(lm["y"] * h))
        status = data.get("status", "normal")

        if status == "normal" or status == "dentro_de_referencia":
            color = (80, 230, 110)
        elif status == "limitado" or status == "fuera_de_referencia":
            color = (60, 155, 255)
        elif status == "excedido":
            color = (60, 70, 255)
        else:
            color = (200, 200, 200)

        angle_text = f"{data['angle']:.0f}"
        tx, ty = pt[0] + 12, pt[1] - 8
        
        box_w, box_h = 32, 16
        rect_x1, rect_y1 = tx - 2, ty - 12
        
        # Anti-overlap logic (Task 9)
        for _ in range(6):
            overlap = False
            rect_x2, rect_y2 = rect_x1 + box_w, rect_y1 + box_h
            for (bx1, by1, bx2, by2) in drawn_boxes:
                if not (rect_x2 < bx1 or rect_x1 > bx2 or rect_y2 < by1 or rect_y1 > by2):
                    overlap = True
                    break
            if not overlap:
                break
            # Try moving it down if it overlaps
            rect_y1 += 18
            ty += 18
            
        drawn_boxes.append((rect_x1, rect_y1, rect_x1 + box_w, rect_y1 + box_h))

        # Background semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x1 + box_w, rect_y1 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x1 + box_w, rect_y1 + box_h), color, 1)
        cv2.putText(frame, angle_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


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
