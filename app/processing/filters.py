import numpy as np
from scipy import signal
from typing import List, Optional

def interpolate_gaps(data: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """
    Interpola linealmente valores NaN en una serie 1D temporal.
    Si el hueco es mayor a max_gap, no lo interpola (lo deja como NaN).
    """
    if len(data) == 0:
        return data
        
    nans = np.isnan(data)
    if not np.any(nans):
        return data
        
    x = np.arange(len(data))
    
    # Encontrar segmentos contiguos de NaNs
    nan_indices = np.where(nans)[0]
    if len(nan_indices) == 0:
        return data
        
    # Segmentación básica para respetar max_gap
    gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    
    data_out = data.copy()
    for gap in gaps:
        if len(gap) <= max_gap:
            # Puntos válidos alrededor del gap para interpolar
            idx = np.where(~np.isnan(data_out))[0]
            if len(idx) > 1:
                data_out[gap] = np.interp(gap, idx, data_out[idx])
                
    return data_out

def apply_savgol_filter(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Aplica un filtro Savitzky-Golay para suavizar una serie temporal 1D.
    Ideal para post-procesamiento (elimina ruido sin cambiar picos de forma tan severa).
    """
    if len(data) < window_length:
        return data
    
    # Interpolar gaps antes de filtrar (savgol no maneja NaNs directamente bien)
    clean_data = interpolate_gaps(data, max_gap=10)
    
    # Si sigue habiendo NaNs, no podemos filtrar esa parte limpiamente
    mask = ~np.isnan(clean_data)
    
    result = clean_data.copy()
    if np.sum(mask) >= window_length:
        # Extraemos solo los válidos continuos (simplificación para este prototipo)
        # En una versión robusta, se filtran los segmentos por separado
        valid_indices = np.where(mask)[0]
        segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
        for seg in segments:
            if len(seg) >= window_length:
                result[seg] = signal.savgol_filter(clean_data[seg], window_length, polyorder)
                
    return result
