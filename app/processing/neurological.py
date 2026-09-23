"""
Analizador Biomecánico y Neurológico (Post-procesamiento).
Aplica análisis de señales (FFT, Peak Detection, Derivadas) a los datos recolectados.
"""

import numpy as np
from scipy import signal
from typing import List, Dict

class NeurologicalAnalyzer:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze_session(self, history: List[Dict]) -> Dict:
        """
        Toma el historial de todos los frames grabados durante una sesión (mano)
        y extrae marcadores clínicos avanzados.
        """
        if len(history) < self.fps * 2: # Mínimo 2 segundos de datos
            return {"error": "Sesión demasiado corta para análisis neurológico (mínimo 2 seg)."}
            
        report = {}
        
        # Extraer series de tiempo
        # Suponemos que cada elemento en history es un frame grabado.
        # Necesitamos reconstruir la posición X,Y de algunos landmarks.
        # Como history en el código original guardaba solo "angles", necesitamos modificar main.py
        # para que pase las posiciones raw o reconstruirlas.
        # Para el Finger Tapping, requerimos la distancia pulgar-índice.
        # Para el Temblor, la posición de la muñeca.
        
        # Debido a que `history` actualmente solo tiene "angles" y "metrics", 
        # requerimos que main.py agregue "raw_landmarks" a cada entrada de `all_measurements`.
        
        raw_available = "raw_landmarks" in history[0]
        if not raw_available:
            return {"error": "Faltan posiciones crudas (raw_landmarks) en el historial."}
            
        # 1. Análisis de Finger Tapping (Bradicinesia)
        tapping_dist = []
        wrist_y = []
        index_tip_y = []
        
        for frame in history:
            lms = frame["raw_landmarks"]
            # Distancia pulgar (4) - índice (8)
            thumb = lms[4]
            index = lms[8]
            dist = np.sqrt((thumb["x"] - index["x"])**2 + (thumb["y"] - index["y"])**2)
            tapping_dist.append(dist)
            
            wrist_y.append(lms[0]["y"])
            index_tip_y.append(lms[8]["y"])
            
        report["finger_tapping"] = self._analyze_tapping(np.array(tapping_dist))
        report["temblor_muneca"] = self._analyze_tremor(np.array(wrist_y))
        report["fluidez_movimiento"] = self._analyze_jerk(np.array(index_tip_y))
        
        return report
        
    def _analyze_tapping(self, dist_array: np.ndarray) -> Dict:
        # Suavizar señal
        smoothed = signal.savgol_filter(dist_array, window_length=11, polyorder=3)
        
        # Encontrar picos (manos abiertas)
        peaks, properties = signal.find_peaks(smoothed, distance=self.fps*0.2, prominence=0.02)
        # Encontrar valles (manos cerradas/taps) invertiendo la señal
        valleys, _ = signal.find_peaks(-smoothed, distance=self.fps*0.2, prominence=0.02)
        
        if len(peaks) < 2:
            return {"status": "Insuficientes taps detectados."}
            
        # Frecuencia
        total_time = len(dist_array) * self.dt
        freq = len(peaks) / total_time
        
        # Decremento de amplitud (fatiga de Parkinson)
        amplitudes = smoothed[peaks]
        slope = 0
        if len(amplitudes) > 2:
            x = np.arange(len(amplitudes))
            slope, _ = np.polyfit(x, amplitudes, 1)
            
        return {
            "frecuencia_hz": round(freq, 2),
            "total_taps": len(peaks),
            "amplitud_promedio": round(np.mean(amplitudes), 3),
            "fatiga_pendiente": round(slope, 5) # Negativo indica que la amplitud decrece con el tiempo
        }

    def _analyze_tremor(self, pos_array: np.ndarray) -> Dict:
        # Quitar tendencia lineal
        detrended = signal.detrend(pos_array)
        
        # FFT para encontrar frecuencia dominante
        freqs, psd = signal.welch(detrended, fs=self.fps, nperseg=min(256, len(pos_array)))
        
        # Filtrar banda de temblor humano (3 - 12 Hz)
        mask = (freqs >= 3.0) & (freqs <= 12.0)
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        
        if len(band_psd) == 0:
            return {"status": "No se detectó señal en banda de temblor."}
            
        dom_idx = np.argmax(band_psd)
        dom_freq = band_freqs[dom_idx]
        dom_power = band_psd[dom_idx]
        
        return {
            "frecuencia_dominante_hz": round(float(dom_freq), 1),
            "poder_amplitud": round(float(dom_power), 6)
        }
        
    def _analyze_jerk(self, pos_array: np.ndarray) -> Dict:
        """
        Derivadas para calcular Jerk (Fluidez).
        """
        # Pos -> Vel -> Acc -> Jerk
        vel = np.gradient(pos_array, self.dt)
        acc = np.gradient(vel, self.dt)
        jerk = np.gradient(acc, self.dt)
        
        # Normalized Jerk (SPARC o RMS)
        mean_squared_jerk = np.mean(jerk**2)
        
        return {
            "velocidad_maxima": round(float(np.max(np.abs(vel))), 3),
            "jerk_rms": round(float(np.sqrt(mean_squared_jerk)), 3) # Un valor alto indica movimiento espástico/robótico
        }
