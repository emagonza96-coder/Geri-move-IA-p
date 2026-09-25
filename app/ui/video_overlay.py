import cv2
import numpy as np

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

class VideoOverlay:
    """Maneja el dibujado sobre el frame original del video (esqueleto, etc)."""
    
    def __init__(self):
        pass
        
    def _draw_angles(self, frame: np.ndarray, landmarks: list, angles: dict):
        h, w = frame.shape[:2]
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
            if vertex_idx >= len(landmarks):
                continue
                
            lm = landmarks[vertex_idx]
            if lm.get("visibility", 0) < 0.5:
                continue

            pt = (int(lm["x"] * w), int(lm["y"] * h))
            status = data.get("status", "normal")

            if status in ("normal", "dentro_de_referencia"):
                color = (80, 230, 110)
            elif status in ("limitado", "fuera_de_referencia"):
                color = (60, 155, 255)
            elif status == "excedido":
                color = (60, 70, 255)
            else:
                color = (200, 200, 200)

            angle_text = f"{data['angle']:.0f}"
            tx, ty = pt[0] + 12, pt[1] - 8
            
            box_w, box_h = 32, 16
            rect_x1, rect_y1 = tx - 2, ty - 12
            
            # Lógica anti-overlap
            for _ in range(6):
                overlap = False
                rect_x2, rect_y2 = rect_x1 + box_w, rect_y1 + box_h
                for (bx1, by1, bx2, by2) in drawn_boxes:
                    if not (rect_x2 < bx1 or rect_x1 > bx2 or rect_y2 < by1 or rect_y1 > by2):
                        overlap = True
                        break
                if not overlap:
                    break
                rect_y1 += 18
                ty += 18
                
            drawn_boxes.append((rect_x1, rect_y1, rect_x1 + box_w, rect_y1 + box_h))

            # Fondo semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x1 + box_w, rect_y1 + box_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x1 + box_w, rect_y1 + box_h), color, 1)
            cv2.putText(frame, angle_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def draw_skeleton(self, frame: np.ndarray, landmarks: list, angles: dict = None, panel_hidden: bool = False) -> np.ndarray:
        """Dibuja el esqueleto y, opcionalmente, los ángulos."""
        h, w = frame.shape[:2]
        
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

        if angles and panel_hidden:
            self._draw_angles(frame, landmarks, angles)
            
        return frame
