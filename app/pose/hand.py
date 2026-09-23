"""
Detector de articulaciones de la mano usando MediaPipe Tasks API (HandLandmarker).
Detecta 21 puntos por mano con coordenadas normalizadas x, y, z.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List, Dict
import os
import urllib.request


HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_hand_model() -> str:
    """Descarga el modelo de manos si no existe localmente y retorna la ruta."""
    model_filename = "hand_landmarker.task"
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_filename)

    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[INFO] Descargando modelo de manos: {model_filename}...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[INFO] Modelo descargado: {size_mb:.1f} MB")

    return model_path


class HandDetector:
    """Detector de landmarks de la mano usando MediaPipe HandLandmarker."""

    # Nombres de los 21 landmarks de la mano
    LANDMARK_NAMES = {
        0:  "WRIST",
        1:  "THUMB_CMC",    2:  "THUMB_MCP",    3:  "THUMB_IP",     4:  "THUMB_TIP",
        5:  "INDEX_MCP",    6:  "INDEX_PIP",    7:  "INDEX_DIP",    8:  "INDEX_TIP",
        9:  "MIDDLE_MCP",   10: "MIDDLE_PIP",   11: "MIDDLE_DIP",   12: "MIDDLE_TIP",
        13: "RING_MCP",     14: "RING_PIP",     15: "RING_DIP",     16: "RING_TIP",
        17: "PINKY_MCP",    18: "PINKY_PIP",    19: "PINKY_DIP",    20: "PINKY_TIP",
    }

    # Conexiones del esqueleto de la mano con color por dedo (BGR, grosor)
    _C_THUMB  = (180, 105, 255)   # Rosa
    _C_INDEX  = (230, 155,  90)   # Azul
    _C_MIDDLE = (200, 200,   0)   # Cian
    _C_RING   = ( 90, 215,  90)   # Verde
    _C_PINKY  = ( 80, 205, 220)   # Amarillo
    _C_PALM   = (180, 180, 180)   # Gris (palma)

    HAND_CONNECTIONS_COLORED = [
        # Palma (muñeca a cada base de dedo)
        ((0, 1),  _C_PALM,  2), ((0, 5),  _C_PALM,  2),
        ((0, 9),  _C_PALM,  2), ((0, 13), _C_PALM,  2),
        ((0, 17), _C_PALM,  2),
        # Conexiones entre bases de dedos
        ((5, 9),  _C_PALM,  1), ((9, 13), _C_PALM,  1), ((13, 17), _C_PALM, 1),
        # Pulgar
        ((1, 2),  _C_THUMB, 2), ((2, 3),  _C_THUMB, 2), ((3, 4),   _C_THUMB, 2),
        # Índice
        ((5, 6),  _C_INDEX, 2), ((6, 7),  _C_INDEX, 2), ((7, 8),   _C_INDEX, 2),
        # Medio
        ((9, 10), _C_MIDDLE, 2), ((10, 11), _C_MIDDLE, 2), ((11, 12), _C_MIDDLE, 2),
        # Anular
        ((13, 14), _C_RING, 2), ((14, 15), _C_RING, 2), ((15, 16), _C_RING, 2),
        # Meñique
        ((17, 18), _C_PINKY, 2), ((18, 19), _C_PINKY, 2), ((19, 20), _C_PINKY, 2),
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
    ):
        """
        Inicializa el detector de manos.

        Args:
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para tracking
            num_hands: Número máximo de manos a detectar (1 o 2)
        """
        model_path = ensure_hand_model()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_hands=num_hands,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def detect(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detecta manos en un frame y retorna landmarks.

        Returns:
            Lista de dicts, uno por mano detectada. Cada dict contiene:
            - "handedness": "Left" o "Right"
            - "landmarks": lista de 21 dicts {x, y, z, visibility}
            Retorna None si no se detectaron manos.
        """
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        hands = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            # Determinar lateralidad (MediaPipe la invierte: "Left" en imagen = mano derecha real)
            handedness = "Right"
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name

            landmarks = []
            for lm in hand_lms:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": 1.0,  # HandLandmarker no reporta visibility, asumimos 1.0
                })

            hands.append({
                "handedness": handedness,
                "landmarks": landmarks,
            })

        return hands

    def draw_hand(
        self,
        frame: np.ndarray,
        hands: List[Dict],
        cam_w: int = 0,
        deformities: Optional[List] = None,
    ) -> np.ndarray:
        """
        Dibuja SOLO el esqueleto de la(s) mano(s) sobre la zona de cámara.
        - cam_w: ancho real de la cámara. Si 0, usa el frame completo.
        - deformities: lista de strings con alertas clínicas; marca círculos rojos.
        Los datos clínicos (goniometría, Kapandji, etc.) son responsabilidad del sidebar.
        """
        h, w, _ = frame.shape
        proj_w = cam_w if cam_w > 0 else w

        for hand_data in hands:
            landmarks = hand_data["landmarks"]
            handedness = hand_data["handedness"]

            # Conexiones coloreadas por dedo
            for (idx_a, idx_b), color, thickness in self.HAND_CONNECTIONS_COLORED:
                lm_a = landmarks[idx_a]
                lm_b = landmarks[idx_b]
                pt_a = (int(lm_a["x"] * proj_w), int(lm_a["y"] * h))
                pt_b = (int(lm_b["x"] * proj_w), int(lm_b["y"] * h))
                cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

            # Puntos de articulación
            for lm in landmarks:
                pt = (int(lm["x"] * proj_w), int(lm["y"] * h))
                cv2.circle(frame, pt, 4, (240, 240, 240), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (0, 200, 175), -1, cv2.LINE_AA)

            # Etiqueta de lateralidad sobre la muñeca
            wrist = landmarks[0]
            wrist_pt = (int(wrist["x"] * proj_w), int(wrist["y"] * h) + 20)
            cv2.putText(frame, f"Mano {handedness}", wrist_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 185), 1, cv2.LINE_AA)

            # Alertas de deformidades (círculos rojos sobre articulaciones)
            if deformities:
                alert_joints = set()
                finger_map = {
                    "indice":  [5, 6, 7, 8],
                    "medio":   [9, 10, 11, 12],
                    "anular":  [13, 14, 15, 16],
                    "menique": [17, 18, 19, 20],
                }
                for d in deformities:
                    fname = d.split(":")[0].strip().lower()
                    for idx in finger_map.get(fname, []):
                        alert_joints.add(idx)
                for idx in alert_joints:
                    lm = landmarks[idx]
                    px, py = int(lm["x"] * proj_w), int(lm["y"] * h)
                    cv2.circle(frame, (px, py), 8, (40, 40, 255), 2, cv2.LINE_AA)

        return frame

    def release(self):
        """Libera recursos de MediaPipe."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
