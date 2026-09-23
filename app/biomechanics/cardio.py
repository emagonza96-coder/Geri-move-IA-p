import time
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import signal

class StepTestAnalyzer:
    """
    Analizador para el 2-Minute Step Test (2MST).
    Cuenta cuántas veces la rodilla supera la altura objetivo.
    """
    def __init__(self):
        self.steps = 0
        self.state_left = "DOWN"
        self.state_right = "DOWN"
        self.start_time = 0.0
        self.active = False
        
    def reset(self):
        self.steps = 0
        self.state_left = "DOWN"
        self.state_right = "DOWN"
        self.start_time = 0.0
        self.active = False
        
    def start(self):
        self.reset()
        self.active = True
        self.start_time = time.time()
        
    def get_elapsed_time(self) -> float:
        if not self.active:
            return 0.0
        return time.time() - self.start_time
        
    def process_frame(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evalúa si hubo un paso en este frame.
        """
        if not self.active:
            return {"steps": self.steps, "active": False, "elapsed": 0.0}
            
        elapsed = self.get_elapsed_time()
        if elapsed > 120.0:  # 2 minutos límite
            self.active = False
            
        try:
            hip_l, hip_r = 23, 24
            knee_l, knee_r = 25, 26
            
            if landmarks[hip_l].get("visibility", 0) > 0.5 and landmarks[knee_l].get("visibility", 0) > 0.5:
                femur_l_len = np.sqrt(
                    (landmarks[hip_l].get("world_x", 0) - landmarks[knee_l].get("world_x", 0))**2 +
                    (landmarks[hip_l].get("world_y", 0) - landmarks[knee_l].get("world_y", 0))**2 +
                    (landmarks[hip_l].get("world_z", 0) - landmarks[knee_l].get("world_z", 0))**2
                )
                
                y_hip_l = landmarks[hip_l].get("world_y", 0)
                y_knee_l = landmarks[knee_l].get("world_y", 0)
                
                target_y_l = y_hip_l + (femur_l_len * 0.5) 
                
                if y_knee_l < target_y_l:
                    if self.state_left == "DOWN":
                        self.state_left = "UP"
                else:
                    if self.state_left == "UP":
                        self.state_left = "DOWN"
                        self.steps += 1
                        
            if landmarks[hip_r].get("visibility", 0) > 0.5 and landmarks[knee_r].get("visibility", 0) > 0.5:
                femur_r_len = np.sqrt(
                    (landmarks[hip_r].get("world_x", 0) - landmarks[knee_r].get("world_x", 0))**2 +
                    (landmarks[hip_r].get("world_y", 0) - landmarks[knee_r].get("world_y", 0))**2 +
                    (landmarks[hip_r].get("world_z", 0) - landmarks[knee_r].get("world_z", 0))**2
                )
                y_hip_r = landmarks[hip_r].get("world_y", 0)
                y_knee_r = landmarks[knee_r].get("world_y", 0)
                
                target_y_r = y_hip_r + (femur_r_len * 0.5)
                
                if y_knee_r < target_y_r:
                    if self.state_right == "DOWN":
                        self.state_right = "UP"
                else:
                    if self.state_right == "UP":
                        self.state_right = "DOWN"
                        self.steps += 1

        except Exception:
            pass
            
        return {
            "steps": self.steps,
            "step_active": self.active,
            "step_elapsed": elapsed,
            "state_l": self.state_left,
            "state_r": self.state_right
        }

class BreathingAnalyzer:
    """
    Analizador de frecuencia respiratoria por movimientos sutiles de los hombros.
    Requiere que el paciente esté estático (sentado o parado).
    """
    def __init__(self, fps: float = 30.0, buffer_seconds: int = 10):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.y_history_l = []
        self.y_history_r = []
        self.current_rpm = 0.0
        
    def reset(self):
        self.y_history_l.clear()
        self.y_history_r.clear()
        self.current_rpm = 0.0
        
    def process_frame(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            sh_l, sh_r = 11, 12
            
            if landmarks[sh_l].get("visibility", 0) > 0.7:
                self.y_history_l.append(landmarks[sh_l].get("world_y", 0))
            if landmarks[sh_r].get("visibility", 0) > 0.7:
                self.y_history_r.append(landmarks[sh_r].get("world_y", 0))
                
            # Mantener buffer
            if len(self.y_history_l) > self.buffer_size:
                self.y_history_l.pop(0)
            if len(self.y_history_r) > self.buffer_size:
                self.y_history_r.pop(0)
                
            # Calcular RPM si hay suficientes datos (ej. al menos 5 segundos)
            if len(self.y_history_l) >= self.fps * 5:
                # Usar promedio de ambos hombros para mitigar ruido
                if len(self.y_history_l) == len(self.y_history_r):
                    signal_data = (np.array(self.y_history_l) + np.array(self.y_history_r)) / 2.0
                    
                    # Quitar tendencia lineal
                    detrended = signal.detrend(signal_data)
                    
                    # FFT
                    freqs, psd = signal.welch(detrended, fs=self.fps, nperseg=min(256, len(detrended)))
                    
                    # Rango normal humano: 10 - 30 respiraciones por minuto (0.16 a 0.5 Hz)
                    mask = (freqs >= 0.15) & (freqs <= 0.6)
                    if np.any(mask):
                        band_freqs = freqs[mask]
                        band_psd = psd[mask]
                        dom_idx = np.argmax(band_psd)
                        dom_freq = band_freqs[dom_idx]
                        
                        self.current_rpm = dom_freq * 60.0
                        
        except Exception:
            pass
            
        return {
            "rpm": round(self.current_rpm, 1) if self.current_rpm > 0 else 0,
            "buffer_fill": min(100, int((len(self.y_history_l) / self.buffer_size) * 100))
        }
