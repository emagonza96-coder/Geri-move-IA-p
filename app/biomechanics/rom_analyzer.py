"""
Módulo para análisis post-sesión de Rangos de Movimiento (ROM).
Detecta repeticiones, extrae máximos y mínimos, y calcula simetría.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Dict, Any, Tuple

class ROMAnalyzer:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        
    def _smooth_signal(self, data: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
        """Suaviza la señal usando el filtro Savitzky-Golay."""
        if len(data) < window_length:
            return data
        return savgol_filter(data, window_length, polyorder)
        
    def detect_repetitions(self, angles: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """
        Detecta repeticiones usando búsqueda de picos.
        Asume que una repetición completa involucra una flexión (o extensión) y un retorno.
        
        Args:
            angles: Lista de ángulos en el tiempo.
            timestamps: Lista de marcas de tiempo correspondientes.
            
        Returns:
            Diccionario con métricas: num_reps, promedios, max/min por repetición.
        """
        if len(angles) < 30: # Muy pocos datos
            return {"num_reps": 0, "error": "Datos insuficientes"}
            
        arr_angles = np.array(angles)
        
        # Suavizar señal
        smoothed = self._smooth_signal(arr_angles)
        
        # Detectar picos (ej. máxima flexión)
        distance = max(int(self.fps * 0.5), 1)
        peaks, properties = find_peaks(smoothed, prominence=15.0, distance=distance)
        
        reps_data = []
        
        for i, peak_idx in enumerate(peaks):
            rep_max = float(smoothed[peak_idx])
            
            # Encontrar el mínimo alrededor de este pico para calcular el rango
            # Límite izquierdo: inicio del arreglo o el pico anterior
            left_bound = 0 if i == 0 else peaks[i-1]
            # Límite derecho: final del arreglo o el siguiente pico
            right_bound = len(smoothed) if i == len(peaks) - 1 else peaks[i+1]
            
            # El mínimo entre el pico actual y el límite izquierdo
            min_left = float(np.min(smoothed[left_bound:peak_idx+1]))
            # El mínimo entre el pico actual y el límite derecho
            min_right = float(np.min(smoothed[peak_idx:right_bound]))
            
            # El mínimo de la repetición es el menor de ambos lados
            rep_min = min(min_left, min_right)
            rom_range = rep_max - rep_min
            
            # Duración aproximada (distancia entre los mínimos locales)
            # Para mayor precisión clínica, podríamos buscar los valles exactos, pero esto es robusto
            duration = 0.0 # Placeholder
            
            reps_data.append({
                "max_angle": round(rep_max, 1),
                "min_angle": round(rep_min, 1),
                "range": round(rom_range, 1),
                "duration_s": duration
            })
            
        num_reps = len(reps_data)
        
        if num_reps == 0:
            return {"num_reps": 0}
            
        # Calcular estadísticas globales
        ranges = [r["range"] for r in reps_data]
        max_angles = [r["max_angle"] for r in reps_data]
        
        return {
            "num_reps": num_reps,
            "avg_range": round(float(np.mean(ranges)), 1),
            "std_range": round(float(np.std(ranges)), 1),
            "avg_max_angle": round(float(np.mean(max_angles)), 1),
            "repetitions": reps_data
        }

    def calculate_symmetry(self, right_val: float, left_val: float) -> float:
        """
        Calcula el Índice de Simetría (SI).
        SI = |D - I| / (0.5 * (D + I)) * 100
        Un valor cercano a 0 indica perfecta simetría.
        """
        denom = 0.5 * (right_val + left_val)
        if denom == 0:
            return 0.0
        si = abs(right_val - left_val) / denom * 100.0
        return round(si, 1)

    def analyze_session(self, all_measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Toma todos los frames guardados en la sesión y extrae las métricas por articulación.
        """
        # Agrupar datos por articulación
        joint_series: Dict[str, Dict[str, list]] = {}
        
        for record in all_measurements:
            # Solo procesar frames del modo cuerpo
            if record.get("mode") != "body":
                continue
                
            ts = record.get("timestamp", 0.0)
            angles_dict = record.get("angles", {})
            
            for joint_key, data in angles_dict.items():
                if data.get("angle") is None:
                    continue
                    
                if joint_key not in joint_series:
                    joint_series[joint_key] = {"angles": [], "timestamps": []}
                    
                joint_series[joint_key]["angles"].append(data["angle"])
                joint_series[joint_key]["timestamps"].append(ts)
                
        # Analizar cada articulación
        report = {}
        for joint_key, series in joint_series.items():
            rep_analysis = self.detect_repetitions(series["angles"], series["timestamps"])
            if rep_analysis.get("num_reps", 0) > 0:
                report[joint_key] = rep_analysis
                
        # Calcular simetría entre pares (ej. hombro_der vs hombro_izq)
        symmetry = {}
        pairs = [
            ("hombro_der", "hombro_izq"),
            ("codo_der", "codo_izq"),
            ("cadera_der", "cadera_izq"),
            ("rodilla_der", "rodilla_izq"),
            ("muneca_der", "muneca_izq")
        ]
        
        for r_key, l_key in pairs:
            if r_key in report and l_key in report:
                r_range = report[r_key]["avg_range"]
                l_range = report[l_key]["avg_range"]
                si = self.calculate_symmetry(r_range, l_range)
                symmetry[f"{r_key}_vs_{l_key}"] = {
                    "symmetry_index_percentage": si,
                    "right_range": r_range,
                    "left_range": l_range
                }
                
        return {
            "version": "1.0",
            "joints_analyzed": report,
            "symmetry_analysis": symmetry
        }
