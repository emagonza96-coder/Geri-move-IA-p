"""
Detector de pose corporal usando MediaPipe Tasks API (PoseLandmarker).
API moderna que reemplaza la legacy mp.solutions.pose.
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


MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

def ensure_model(model_complexity: int = 1) -> str:
    """Descarga el modelo si no existe localmente y retorna la ruta."""
    model_name = ["lite", "full", "heavy"][model_complexity]
    model_filename = f"pose_landmarker_{model_name}.task"
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_filename)
    
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[INFO] Descargando modelo PoseLandmarker {model_name}...")
        url = MODEL_URLS.get(model_complexity, MODEL_URLS[1])
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Modelo descargado en: {model_path}")
        
    return model_path


class PoseDetector:
    """Detector de pose humana usando MediaPipe Tasks PoseLandmarker."""

    # Color BGR por segmento corporal
    _C_TORSO  = (0,   200, 175)   # Cian
    _C_ARM_L  = (230, 155,  90)   # Azul claro
    _C_ARM_R  = (160,  90, 230)   # Violeta
    _C_LEG_L  = ( 90, 215,  90)   # Verde
    _C_LEG_R  = ( 80, 205, 220)   # Amarillo dorado
    _C_NECK   = (210, 180, 240)   # Lila claro (cuello/cervical)

    # Lista de (par_de_landmarks, color_BGR, grosor)
    SKELETON_CONNECTIONS_COLORED = [
        # Torso
        ((11, 12), _C_TORSO, 3), ((11, 23), _C_TORSO, 3),
        ((12, 24), _C_TORSO, 3), ((23, 24), _C_TORSO, 3),
        # Cuello / cervical (oreja → hombro + línea entre orejas)
        ((7,  8),  _C_NECK,  2),
        ((7,  11), _C_NECK,  2), ((8,  12), _C_NECK,  2),
        # Brazo izquierdo
        ((11, 13), _C_ARM_L, 2), ((13, 15), _C_ARM_L, 2),
        ((15, 17), _C_ARM_L, 1), ((15, 19), _C_ARM_L, 1),
        # Brazo derecho
        ((12, 14), _C_ARM_R, 2), ((14, 16), _C_ARM_R, 2),
        ((16, 18), _C_ARM_R, 1), ((16, 20), _C_ARM_R, 1),
        # Pierna izquierda
        ((23, 25), _C_LEG_L, 2), ((25, 27), _C_LEG_L, 2),
        ((27, 29), _C_LEG_L, 1), ((27, 31), _C_LEG_L, 1),
        # Pierna derecha
        ((24, 26), _C_LEG_R, 2), ((26, 28), _C_LEG_R, 2),
        ((28, 30), _C_LEG_R, 1), ((28, 32), _C_LEG_R, 1),
    ]

    # Nombres de landmarks principales
    LANDMARK_NAMES = {
        0:  "nariz",
        7:  "oreja_izq",  8:  "oreja_der",
        11: "hombro_izq", 12: "hombro_der",
        13: "codo_izq",   14: "codo_der",
        15: "muneca_izq", 16: "muneca_der",
        23: "cadera_izq", 24: "cadera_der",
        25: "rodilla_izq",26: "rodilla_der",
        27: "tobillo_izq",28: "tobillo_der",
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Inicializa el detector de pose con MediaPipe Tasks API.

        Args:
            min_detection_confidence: Confianza mínima para detección (0.0-1.0)
            min_tracking_confidence: Confianza mínima para tracking (0.0-1.0)
            model_complexity: 0=lite, 1=full, 2=heavy
        """
        # Descargar modelo si no existe
        model_path = ensure_model(model_complexity)

        # Configurar PoseLandmarker en modo VIDEO
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.time()   # Referencia de tiempo real para timestamps precisos

    def detect(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detecta pose en un frame BGR de OpenCV.

        Args:
            frame: Frame BGR de OpenCV (numpy array)

        Returns:
            Lista de landmarks con x, y, z, visibility o None si no se detecta pose
        """
        # Convertir BGR → RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp basado en tiempo real (ms desde el inicio).
        # Usar tiempo real evita desincronización cuando la cámara corre a FPS variables.
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        # Detectar
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Convertir al formato que usa el resto del sistema
        landmarks = []
        for lm in result.pose_landmarks[0]:  # Primera persona detectada
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })

        return landmarks

    @staticmethod
    def get_torso_visibility(landmarks: List[Dict], threshold: float = 0.5) -> int:
        """
        Cuenta cuántos landmarks de referencia corporal (hombros + caderas) son visibles.
        Se usa para decidir si el cuerpo está suficientemente visible o si la persona
        está demasiado cerca y solo se ve una extremidad.

        Args:
            landmarks: Lista de landmarks de MediaPipe Pose
            threshold: Umbral mínimo de visibilidad (default 0.5)

        Returns:
            Número de puntos de referencia visibles (0 a 4)
        """
        torso_indices = [11, 12, 23, 24]  # hombro_izq, hombro_der, cadera_izq, cadera_der
        return sum(1 for idx in torso_indices if landmarks[idx]["visibility"] > threshold)

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: List[Dict],
        angles: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Dibuja el esqueleto con huesos en color por segmento corporal.
        """
        h, w, _ = frame.shape

        # Huesos con color de segmento
        for (idx_a, idx_b), color, thickness in self.SKELETON_CONNECTIONS_COLORED:
            lm_a = landmarks[idx_a]
            lm_b = landmarks[idx_b]
            if lm_a["visibility"] < 0.5 or lm_b["visibility"] < 0.5:
                continue
            pt_a = (int(lm_a["x"] * w), int(lm_a["y"] * h))
            pt_b = (int(lm_b["x"] * w), int(lm_b["y"] * h))
            cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

        # Articulaciones principales: doble círculo (blanco exterior + cian interior)
        for idx in self.LANDMARK_NAMES:
            lm = landmarks[idx]
            if lm["visibility"] < 0.5:
                continue
            pt = (int(lm["x"] * w), int(lm["y"] * h))
            cv2.circle(frame, pt, 5, (240, 240, 240), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (0, 200, 175), -1, cv2.LINE_AA)

        if angles:
            self._draw_angles(frame, landmarks, angles, w, h)

        return frame

    def _draw_angles(
        self,
        frame: np.ndarray,
        landmarks: List[Dict],
        angles: Dict,
        w: int,
        h: int,
    ):
        """Dibuja el valor del ángulo junto a cada articulación con fondo estimado fijo."""
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
            if lm["visibility"] < 0.5:
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

            # Fondo de tamaño estimado fijo: evita cv2.getTextSize() por frame
            # "180" a escala 0.44 cabe en ~38x16 px
            cv2.rectangle(frame, (tx - 2, ty - 13), (tx + 38, ty + 4), (20, 20, 20), -1)
            cv2.rectangle(frame, (tx - 2, ty - 13), (tx + 38, ty + 4), color, 1)
            cv2.putText(frame, angle_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    def release(self):
        """Libera recursos de MediaPipe."""
        self.detector.close()
