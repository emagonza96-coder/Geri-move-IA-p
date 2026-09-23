import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, Dict

class StereoCalibrator:
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), square_size_cm: float = 2.5):
        """
        Inicializa el calibrador estéreo.
        
        Args:
            checkerboard_size: Número de intersecciones internas (columnas, filas) del tablero de ajedrez.
            square_size_cm: Tamaño físico del lado de un cuadrado en centímetros.
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_cm = square_size_cm
        
        # Criterios de terminación para la calibración
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Preparar los puntos 3D del mundo real
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_cm
        
        # Arreglos para almacenar puntos
        self.objpoints = [] # Puntos 3D en el espacio del mundo real
        self.imgpoints_left = [] # Puntos 2D en el plano de la cámara izquierda
        self.imgpoints_right = [] # Puntos 2D en el plano de la cámara derecha
        
    def add_image_pair(self, img_left: np.ndarray, img_right: np.ndarray) -> bool:
        """
        Intenta encontrar el tablero de ajedrez en ambas imágenes y agrega los puntos si es exitoso.
        
        Returns:
            True si se encontró en ambas imágenes, False en caso contrario.
        """
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.checkerboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.checkerboard_size, None)
        
        if ret_left and ret_right:
            self.objpoints.append(self.objp)
            
            # Refinar esquinas
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
            
            self.imgpoints_left.append(corners2_left)
            self.imgpoints_right.append(corners2_right)
            return True
            
        return False
        
    def draw_corners(self, img_left: np.ndarray, img_right: np.ndarray, success: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Dibuja las esquinas detectadas sobre las imágenes para feedback visual."""
        res_left = img_left.copy()
        res_right = img_right.copy()
        
        if len(self.imgpoints_left) > 0 and success:
            cv2.drawChessboardCorners(res_left, self.checkerboard_size, self.imgpoints_left[-1], True)
            cv2.drawChessboardCorners(res_right, self.checkerboard_size, self.imgpoints_right[-1], True)
            
        return res_left, res_right
        
    def calibrate(self, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Realiza la calibración estéreo usando los puntos recopilados.
        image_shape: (width, height)
        """
        if len(self.objpoints) < 5:
            print("[ERROR] Se necesitan al menos 5 pares de imágenes con el tablero detectado.")
            return None
            
        print(f"[INFO] Calibrando con {len(self.objpoints)} pares de imágenes...")
        
        # Calibración individual inicial
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_shape, None, None)
            
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_shape, None, None)
            
        # Calibración estéreo
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_l, dist_l, mtx_r, dist_r, image_shape,
            criteria=self.criteria, flags=flags)
            
        # Calcular matrices de proyección para triangulación
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            M1, d1, M2, d2, image_shape, R, T)
            
        calibration_data = {
            "error_rms": ret,
            "proj_mat_left": P1.tolist(),
            "proj_mat_right": P2.tolist(),
            "cam_mat_left": M1.tolist(),
            "cam_mat_right": M2.tolist(),
            "dist_left": d1.tolist(),
            "dist_right": d2.tolist(),
            "rotation": R.tolist(),
            "translation": T.tolist()
        }
        
        print(f"[INFO] Calibración completa. Error RMS: {ret:.4f}")
        return calibration_data
        
    def save_calibration(self, data: Dict, filepath: str):
        """Guarda la matriz de calibración en un archivo JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Calibración guardada en {filepath}")
        
    @staticmethod
    def load_calibration(filepath: str) -> Optional[Dict]:
        """Carga la matriz de calibración desde un archivo JSON."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convertir listas de vuelta a numpy arrays
        for key in data:
            if isinstance(data[key], list):
                data[key] = np.array(data[key])
        return data
