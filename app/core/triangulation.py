import cv2
import numpy as np
from typing import Dict, List, Tuple

class Triangulator:
    def __init__(self, calibration_data: Dict):
        """
        Inicializa el motor de triangulación 3D.
        
        Args:
            calibration_data: Diccionario con P1 (proj_mat_left) y P2 (proj_mat_right) 
                              cargado desde el archivo de calibración.
        """
        self.P1 = np.array(calibration_data['proj_mat_left'])
        self.P2 = np.array(calibration_data['proj_mat_right'])
        
    def triangulate_point(self, pt_left: Tuple[float, float], pt_right: Tuple[float, float]) -> np.ndarray:
        """
        Triangula un solo punto 2D encontrado en la cámara izquierda y derecha 
        hacia un punto 3D en el mundo real.
        
        Args:
            pt_left: Coordenadas (x, y) en la imagen izquierda (pixeles)
            pt_right: Coordenadas (x, y) en la imagen derecha (pixeles)
            
        Returns:
            np.ndarray de forma (3,) con coordenadas [X, Y, Z] en el mundo físico.
        """
        pts_left = np.array([[pt_left]], dtype=np.float32)
        pts_right = np.array([[pt_right]], dtype=np.float32)
        
        # triangulatePoints requiere forma (2, N), y devuelve (4, N) puntos homogéneos
        pts4d = cv2.triangulatePoints(
            self.P1, self.P2, 
            pts_left.reshape(2, 1), 
            pts_right.reshape(2, 1)
        )
        
        # Convertir de coordenadas homogéneas (X, Y, Z, W) a Euclidianas (X/W, Y/W, Z/W)
        pts3d = pts4d[:3, :] / pts4d[3, :]
        
        return pts3d.reshape(3)
        
    def triangulate_landmarks(self, landmarks_left: List[Dict], landmarks_right: List[Dict], width: int, height: int) -> List[Dict]:
        """
        Triangula todos los landmarks (que MediaPipe entrega normalizados de 0.0 a 1.0)
        y devuelve una lista de diccionarios con coordenadas 3D reales.
        
        Solo triangula si la visibilidad es mayor a un umbral en ambas cámaras.
        """
        landmarks_3d = []
        
        for lm_l, lm_r in zip(landmarks_left, landmarks_right):
            lm_3d = {
                "visibility": min(lm_l["visibility"], lm_r["visibility"])
            }
            
            # Si el punto es visible en ambas cámaras (confianza > 50%)
            if lm_3d["visibility"] >= 0.5:
                # Convertir coordenadas normalizadas a pixeles
                px_l = (lm_l["x"] * width, lm_l["y"] * height)
                px_r = (lm_r["x"] * width, lm_r["y"] * height)
                
                pt_3d = self.triangulate_point(px_l, px_r)
                lm_3d["x"] = pt_3d[0]
                lm_3d["y"] = pt_3d[1]
                lm_3d["z"] = pt_3d[2]
            else:
                lm_3d["x"] = None
                lm_3d["y"] = None
                lm_3d["z"] = None
                
            landmarks_3d.append(lm_3d)
            
        return landmarks_3d
