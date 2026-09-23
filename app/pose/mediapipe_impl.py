import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List, Dict, Any
import os
import urllib.request

from .base import PoseEstimator

MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

def ensure_model(model_complexity: int = 1) -> str:
    """Descarga el modelo si no existe localmente y retorna la ruta."""
    model_name = ["lite", "full", "heavy"][model_complexity]
    model_filename = f"pose_landmarker_{model_name}.task"
    # Ahora la ruta relativa a este archivo sube dos niveles (app/pose -> app -> root/models)
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", model_filename)
    
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[INFO] Descargando modelo PoseLandmarker {model_name}...")
        url = MODEL_URLS.get(model_complexity, MODEL_URLS[1])
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Modelo descargado en: {model_path}")
        
    return model_path

class MediaPipePoseEstimator(PoseEstimator):
    """Implementación de PoseEstimator usando MediaPipe Tasks."""

    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5, model_complexity: int = 1):
        model_path = ensure_model(model_complexity)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: float) -> Optional[List[Dict[str, Any]]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe exige que el timestamp sea estrictamente entero y creciente.
        ts_int = int(timestamp_ms)
        
        result = self.detector.detect_for_video(mp_image, ts_int)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = []
        # Extraemos tanto 2D (normalizado) como 3D (metros)
        pose_2d = result.pose_landmarks[0]
        pose_3d = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        for idx, lm in enumerate(pose_2d):
            lm_data = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            if pose_3d:
                lm_data["world_x"] = pose_3d[idx].x
                lm_data["world_y"] = pose_3d[idx].y
                lm_data["world_z"] = pose_3d[idx].z
            landmarks.append(lm_data)

        return landmarks

    def release(self):
        self.detector.close()
