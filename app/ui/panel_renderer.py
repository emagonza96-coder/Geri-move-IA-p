import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from app.config.loader import get_ui_config
import time

class PanelRenderer:
    def __init__(self, width: int = 240, height: int = 720):
        self.width = width
        self.height = height
        self.bg_color = (14, 14, 18) # BGR
        self.pad = 10
        
        ui_conf = get_ui_config()
        c = ui_conf.get("clock_color", [180, 180, 195])
        self.clock_color = (c[0], c[1], c[2])

        # Cargar fuentes
        font_dir = Path("app/ui/fonts")
        try:
            self.font_title = ImageFont.truetype(str(font_dir / "Roboto-Bold.ttf"), 16)
            self.font_header = ImageFont.truetype(str(font_dir / "Roboto-Medium.ttf"), 12)
            self.font_normal = ImageFont.truetype(str(font_dir / "Roboto-Regular.ttf"), 11)
            self.font_small = ImageFont.truetype(str(font_dir / "Roboto-Regular.ttf"), 10)
            self.font_tiny = ImageFont.truetype(str(font_dir / "Roboto-Regular.ttf"), 9)
            self.font_large = ImageFont.truetype(str(font_dir / "Roboto-Bold.ttf"), 22)
        except Exception as e:
            print(f"[WARN] No se pudieron cargar las fuentes Roboto: {e}. Usando default.")
            self.font_title = ImageFont.load_default()
            self.font_header = ImageFont.load_default()
            self.font_normal = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_tiny = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def _bgr_to_rgb(self, color):
        return (color[2], color[1], color[0]) if len(color) == 3 else color

    def render(
        self,
        angles: dict,
        fps: float,
        frame_num: int,
        cached_ts: str = "",
        active_mode: str = "BODY",
        hand_metrics: dict = None,
        session_info: dict = None,
        quality_indicator: float = 100.0,
        is_recording: bool = False,
    ) -> np.ndarray:
        
        # Crear imagen PIL RGB con color de fondo (usamos RGB directo convirtiendo bg_color)
        bg_rgb = self._bgr_to_rgb(self.bg_color)
        img = Image.new("RGB", (self.width, self.height), bg_rgb)
        draw = ImageDraw.Draw(img)
        w, h = self.width, self.height
        px0 = 0

        # Línea separadora izquierda
        draw.line([(0, 0), (0, h)], fill=self._bgr_to_rgb((0, 195, 165)), width=1)

        # ── ENCABEZADO ──
        HEADER_H = 75
        draw.rectangle([(px0, 0), (w, HEADER_H)], fill=self._bgr_to_rgb((20, 20, 26)))
        draw.line([(px0, HEADER_H), (w, HEADER_H)], fill=self._bgr_to_rgb((0, 185, 155)), width=1)

        draw.text((px0 + self.pad, 8), "MOBILITY SCAN", font=self.font_title, fill=self._bgr_to_rgb((0, 220, 190)))
        draw.text((px0 + self.pad, 26), cached_ts, font=self.font_normal, fill=self._bgr_to_rgb(self.clock_color))

        if not session_info:
            session_info = {"id": "Anon-1", "task": "Libre", "intent": 1, "view": "Frontal", "timer": "00:00"}
        
        sid, task = session_info.get('id', 'N/A'), session_info.get('task', 'N/A')
        intent, view = session_info.get('intent', 1), session_info.get('view', 'Frontal')
        timer = session_info.get('timer', '00:00')
        
        draw.text((px0 + self.pad, 42), f"ID: {sid} | Tarea: {task} (#{intent})", font=self.font_small, fill=self._bgr_to_rgb((180, 180, 180)))
        draw.text((px0 + self.pad, 56), f"Vista: {view} | Timer: {timer}", font=self.font_small, fill=self._bgr_to_rgb((180, 180, 180)))

        # FPS y Q
        fps_color = (80, 240, 100) if fps >= 20 else (60, 165, 255) if fps >= 10 else (60, 80, 255)
        draw.text((w - 45, 10), f"{fps:.0f}fps", font=self.font_small, fill=self._bgr_to_rgb(fps_color))
        
        qual_color = (80, 240, 100) if quality_indicator >= 85 else (0, 140, 255) if quality_indicator >= 50 else (60, 60, 255)
        draw.text((w - 55, 26), f"Q: {quality_indicator:.0f}%", font=self.font_small, fill=self._bgr_to_rgb(qual_color))

        # Badge
        mode_color = (255, 140, 80) if active_mode == "HAND" else (0, 195, 165)
        mode_label = "MANO" if active_mode == "HAND" else "CUERPO"
        draw.rectangle([(w - 55, 44), (w - 4, 60)], fill=self._bgr_to_rgb((30, 30, 36)))
        draw.text((w - 50, 46), mode_label, font=self.font_small, fill=self._bgr_to_rgb(mode_color))

        cy = HEADER_H + 12

        # ── CUERPO ──
        if active_mode == "BODY" and angles:
            PANEL_W = w - self.pad * 2
            draw.text((px0 + self.pad, cy), "ANGULOS ROM", font=self.font_header, fill=self._bgr_to_rgb((0, 200, 175)))
            draw.line([(px0 + self.pad, cy + 16), (w - self.pad, cy + 16)], fill=self._bgr_to_rgb((0, 120, 105)), width=1)
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
            LABELS = {
                "cuello_inclinacion": "Cuello Incl", "cuello_rotacion": "Cuello Rot",
                "tronco_inclinacion": "Tronco Incl", "tronco_rotacion": "Tronco Rot",
                "hombro_izq": "Hombro (I)", "hombro_der": "Hombro (D)",
                "codo_izq": "Codo (I)", "codo_der": "Codo (D)",
                "cadera_izq": "Cadera (I)", "cadera_der": "Cadera (D)",
                "rodilla_izq": "Rodilla (I)", "rodilla_der": "Rodilla (D)",
                "muneca_izq": "Muñeca (I)", "muneca_der": "Muñeca (D)"
            }

            COL_W = PANEL_W // 2
            ROW_H = 34
            
            for keys_row in LAYOUT_ORDEN:
                for c_idx, key in enumerate(keys_row):
                    if not key: continue
                    data = angles.get(key, {})
                    cx = px0 + self.pad + (c_idx * COL_W)
                    
                    name = LABELS.get(key, key)
                    angle_val = data.get("angle")
                    status = data.get("status", "fuera_de_cuadro")
                    rom_range = data.get("rom_range")
                    
                    if status in ("normal", "dentro_de_referencia"): color = (80, 230, 110)
                    elif status in ("limitado", "fuera_de_referencia"): color = (60, 155, 255)
                    elif status == "excedido": color = (60, 70, 255)
                    else: color = (110, 110, 110)
                    
                    rgb_color = self._bgr_to_rgb(color)

                    draw.ellipse([(cx + 2, cy + 2), (cx + 8, cy + 8)], fill=rgb_color)
                    draw.text((cx + 14, cy - 2), name[:15], font=self.font_tiny, fill=self._bgr_to_rgb((200, 200, 210)))
                    
                    txt = f"{angle_val:.0f}°" if angle_val is not None else "--"
                    draw.text((cx + COL_W - 28, cy - 2), txt, font=self.font_small, fill=rgb_color)
                    
                    bar_y = cy + 12
                    bar_w = COL_W - 12
                    draw.rectangle([(cx, bar_y), (cx + bar_w, bar_y + 4)], fill=self._bgr_to_rgb((40, 40, 46)))
                    
                    if angle_val is not None and rom_range:
                        r_min, r_max = rom_range
                        if r_max > 0:
                            scale_max = max(r_max * 1.2, 1)
                            x_min = int((r_min / scale_max) * bar_w)
                            x_max = int((r_max / scale_max) * bar_w)
                            
                            draw.rectangle([(cx + x_min, bar_y), (cx + x_max, bar_y + 4)], fill=self._bgr_to_rgb((50, 90, 60)))
                            fill_f = min(angle_val / scale_max, 1.0)
                            fp2 = int(bar_w * fill_f)
                            if fp2 > 0:
                                draw.rectangle([(cx, bar_y), (cx + fp2, bar_y + 4)], fill=rgb_color)
                            
                            draw.line([(cx + x_min, bar_y - 2), (cx + x_min, bar_y + 6)], fill=self._bgr_to_rgb((150, 150, 150)), width=1)
                            draw.line([(cx + x_max, bar_y - 2), (cx + x_max, bar_y + 6)], fill=self._bgr_to_rgb((150, 150, 150)), width=1)
                cy += ROW_H

        # ── MANO ──
        elif active_mode == "HAND" and hand_metrics:
            gonio = hand_metrics.get("goniometria", {})
            abd   = hand_metrics.get("abduccion_pulgar")
            cierre = hand_metrics.get("indice_cierre", 0.0)
            kapandji = hand_metrics.get("kapandji_nivel", 0)
            deformities = hand_metrics.get("deformidades", [])

            draw.text((px0 + self.pad, cy), "GONIOMETRIA MANO", font=self.font_header, fill=self._bgr_to_rgb((255, 160, 80)))
            draw.line([(px0 + self.pad, cy + 16), (w - self.pad, cy + 16)], fill=self._bgr_to_rgb((120, 75, 35)), width=1)
            cy += 24

            finger_rows = [
                ("PULGAR",  [("MCF", gonio.get("pulgar_mcf")), ("IF",  gonio.get("pulgar_if"))]),
                ("INDICE",  [("MCF", gonio.get("indice_mcf")), ("IFP", gonio.get("indice_ifp")), ("IFD", gonio.get("indice_ifd"))]),
                ("MEDIO",   [("MCF", gonio.get("medio_mcf")),  ("IFP", gonio.get("medio_ifp")),  ("IFD", gonio.get("medio_ifd"))]),
                ("ANULAR",  [("MCF", gonio.get("anular_mcf")), ("IFP", gonio.get("anular_ifp")), ("IFD", gonio.get("anular_ifd"))]),
                ("MENIQUE", [("MCF", gonio.get("menique_mcf")),("IFP", gonio.get("menique_ifp")),("IFD", gonio.get("menique_ifd"))]),
            ]
            ROW_H, HEADER_H = 15, 16

            for finger_name, joints in finger_rows:
                draw.text((px0 + self.pad, cy), finger_name, font=self.font_small, fill=self._bgr_to_rgb((170, 170, 185)))
                cy += HEADER_H

                for j_label, j_val in joints:
                    if j_val is None:
                        angle_str, bar_fill, color = "--", 0, (70, 70, 80)
                    else:
                        angle_str = f"{j_val:.0f}°"
                        bar_fill = int(max(0, min(j_val, 120)) / 120 * 70)
                        if j_val < -5 or j_val > 110: color = (60, 60, 240)
                        elif "IFP" in j_label: color = (80, 215, 255)
                        elif "MCF" in j_label: color = (150, 150, 40)
                        else: color = (90, 185, 100)

                    rgb_color = self._bgr_to_rgb(color)
                    draw.text((px0 + self.pad, cy), j_label, font=self.font_tiny, fill=self._bgr_to_rgb((110, 110, 125)))
                    
                    bx_bar, bx_end = px0 + self.pad + 30, px0 + self.pad + 100
                    bar_y = cy + 4
                    draw.rectangle([(bx_bar, bar_y), (bx_end, bar_y + 4)], fill=self._bgr_to_rgb((35, 35, 42)))
                    if bar_fill > 0:
                        draw.rectangle([(bx_bar, bar_y), (bx_bar + bar_fill, bar_y + 4)], fill=rgb_color)
                    draw.text((bx_end + 8, cy), angle_str, font=self.font_tiny, fill=rgb_color)
                    cy += ROW_H

                draw.line([(px0 + self.pad, cy + 2), (w - self.pad, cy + 2)], fill=self._bgr_to_rgb((30, 30, 38)), width=1)
                cy += 6

            draw.line([(px0 + self.pad, cy), (w - self.pad, cy)], fill=self._bgr_to_rgb((0, 140, 120)), width=1)
            cy += 8
            draw.text((px0 + self.pad, cy), "RESUMEN", font=self.font_small, fill=self._bgr_to_rgb((255, 160, 80)))
            cy += 18

            abd_str = f"{abd:.0f}°" if abd is not None else "--"
            draw.text((px0 + self.pad, cy), f"Abd.Pulgar  {abd_str}", font=self.font_small, fill=self._bgr_to_rgb((180, 220, 220)))
            cy += 16

            pct = int((cierre or 0) * 100)
            c_color = (60, 220, 80) if pct > 70 else (60, 130, 255)
            draw.text((px0 + self.pad, cy), "Cierre puno", font=self.font_small, fill=self._bgr_to_rgb((160, 160, 175)))
            bpx = px0 + self.pad + 78
            draw.rectangle([(bpx, cy + 2), (bpx + 50, cy + 10)], fill=self._bgr_to_rgb((35, 35, 42)))
            draw.rectangle([(bpx, cy + 2), (bpx + int(50 * pct / 100), cy + 10)], fill=self._bgr_to_rgb(c_color))
            draw.text((bpx + 56, cy), f"{pct}%", font=self.font_small, fill=self._bgr_to_rgb(c_color))
            cy += 16

            kap_color = (60, 220, 80) if kapandji >= 8 else (80, 200, 200) if kapandji > 4 else (60, 130, 255)
            draw.text((px0 + self.pad, cy), f"Kapandji    {kapandji}/10", font=self.font_small, fill=self._bgr_to_rgb(kap_color))
            cy += 16

            if deformities:
                for d in deformities[:3]:
                    draw.text((px0 + self.pad, cy), f"! {d[:25]}", font=self.font_tiny, fill=self._bgr_to_rgb((60, 60, 240)))
                    cy += 14

        # ── FOOTER ESTADO DE GRABACIÓN ──
        ph = self.height
        pw = self.width
        if is_recording:
            draw.rectangle([(0, ph - 60), (pw, ph)], fill=self._bgr_to_rgb((15, 15, 15)))
            draw.text((10, ph - 50), "GRABANDO SESION", font=self.font_header, fill=self._bgr_to_rgb((0, 0, 255)))
            draw.text((10, ph - 30), "[Q] Detener [P] Pausa", font=self.font_small, fill=self._bgr_to_rgb((150, 150, 150)))
            if time.time() % 1.0 > 0.4:
                draw.ellipse([(pw - 25, ph - 45), (pw - 15, ph - 35)], fill=self._bgr_to_rgb((0, 0, 255)))
            draw.rectangle([(0, 0), (pw-1, ph-1)], outline=self._bgr_to_rgb((0, 0, 255)), width=2)
        else:
            draw.rectangle([(0, ph - 85), (pw, ph)], fill=self._bgr_to_rgb((15, 15, 15)))
            draw.text((10, ph - 75), "MODO CALIBRACION", font=self.font_header, fill=self._bgr_to_rgb((0, 255, 255)))
            draw.text((10, ph - 55), "NO GRABANDO", font=self.font_header, fill=self._bgr_to_rgb((0, 255, 255)))
            draw.text((10, ph - 30), "[Q] Iniciar [O] Forzar", font=self.font_small, fill=self._bgr_to_rgb((150, 150, 150)))

        # Convertir de RGB PIL a BGR OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
