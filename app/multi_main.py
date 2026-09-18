"""
Sistema Multi-Cámara 3D para Detección de Movilidad
===================================================
Utiliza dos cámaras y triangulación estéreo para calcular ángulos
articulares en el espacio 3D físico real.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.pose_detector import PoseDetector
from core.angle_calculator import calculate_all_angles
from core.multi_camera import MultiCameraManager
from core.triangulation import Triangulator
from core.calibration import StereoCalibrator


def parse_args():
    parser = argparse.ArgumentParser(description="Mobility Scan Multi-Cámara 3D")
    parser.add_argument("--source1", default="0", help="Fuente de cámara izquierda")
    parser.add_argument("--source2", default="1", help="Fuente de cámara derecha")
    parser.add_argument("--calib", default="calibration.json", help="Archivo JSON de calibración")
    parser.add_argument("--calibrate-mode", action="store_true", help="Inicia en modo captura de tablero de ajedrez")
    parser.add_argument("--output-dir", default="output", help="Carpeta de resultados")
    return parser.parse_args()


def process_calibration_mode(manager: MultiCameraManager, calib_file: str):
    calibrator = StereoCalibrator(checkerboard_size=(9, 6), square_size_cm=2.5)
    print("[INFO] Modo de Calibración iniciado.")
    print("[INFO] Muestra el tablero de ajedrez a ambas cámaras.")
    print("[INFO] Presiona 'C' para capturar un par. Necesitas al menos 5 pares (10-15 recomendado).")
    print("[INFO] Presiona 'Q' para terminar de capturar y procesar la calibración.")
    
    manager.start()
    
    while True:
        success, frames = manager.read()
        if not success or frames[0] is None or frames[1] is None:
            continue
            
        frame_l, frame_r = frames[0], frames[1]
        
        # Mostrar las vistas
        combined = np.hstack((frame_l, frame_r))
        cv2.putText(combined, f"Pares capturados: {len(calibrator.objpoints)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Calibracion Stereo (Izq | Der)", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("[INFO] Intentando detectar tablero...")
            found = calibrator.add_image_pair(frame_l, frame_r)
            if found:
                print(f"[EXITO] Tablero detectado. Total pares: {len(calibrator.objpoints)}")
            else:
                print("[ERROR] No se detectó tablero en una o ambas cámaras.")
                
        elif key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    
    if len(calibrator.objpoints) >= 5:
        h, w = frames[0].shape[:2]
        data = calibrator.calibrate((w, h))
        if data:
            calibrator.save_calibration(data, calib_file)
            print("[INFO] Calibración exitosa. Ahora puedes ejecutar el sistema normal.")
    else:
        print("[ERROR] No se capturaron suficientes pares válidos para calibrar.")
        
    manager.release()
    sys.exit(0)


def main():
    args = parse_args()
    
    # Procesar fuentes
    src1 = int(args.source1) if args.source1.isdigit() else args.source1
    src2 = int(args.source2) if args.source2.isdigit() else args.source2
    
    manager = MultiCameraManager([src1, src2])
    
    if args.calibrate_mode:
        process_calibration_mode(manager, args.calib)
        
    # Modo Inferencia 3D
    calib_data = StereoCalibrator.load_calibration(args.calib)
    if not calib_data:
        print(f"[ERROR] Archivo de calibración no encontrado: {args.calib}")
        print("[HINT] Ejecuta primero con --calibrate-mode para crearlo.")
        manager.release()
        sys.exit(1)
        
    triangulator = Triangulator(calib_data)
    
    # Inicializar detectores 2D (uno por vista)
    detector_l = PoseDetector(model_complexity=1)
    detector_r = PoseDetector(model_complexity=1)
    
    manager.start()
    
    print("[INFO] Sistema Multi-Cámara 3D iniciado. Presiona Q para salir.")
    
    while True:
        success, frames = manager.read()
        if not success or frames[0] is None or frames[1] is None:
            continue
            
        frame_l, frame_r = frames[0], frames[1]
        
        # 1. Detección 2D independiente
        res_l, lm_l = detector_l.detect_pose(frame_l)
        res_r, lm_r = detector_r.detect_pose(frame_r)
        
        # 2. Triangulación 3D
        if lm_l and lm_r:
            h, w = frame_l.shape[:2]
            landmarks_3d = triangulator.triangulate_landmarks(lm_l, lm_r, w, h)
            
            # 3. Cálculo de Ángulos 3D
            angles_3d = calculate_all_angles(landmarks_3d)
            
            # TODO: Guardar datos a JSON, suavizar, dibujar HUD avanzado
            # Por ahora, simplemente dibujar esqueleto 2D en ambas cámaras para visualización
            if res_l is not None: frame_l = res_l
            if res_r is not None: frame_r = res_r
            
            # Añadir info debug
            cv2.putText(frame_l, "Camara Izq", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_r, "Camara Der", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # Unir visualmente ambas cámaras
        h1, w1 = frame_l.shape[:2]
        h2, w2 = frame_r.shape[:2]
        
        # Escalar si son diferentes
        if h1 != h2 or w1 != w2:
            frame_r = cv2.resize(frame_r, (w1, h1))
            
        combined = np.hstack((frame_l, frame_r))
        cv2.imshow("Mobility Scan 3D", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    manager.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
