"""
Motor de cálculo de métricas espaciales para la mano.
Evalúa la flexibilidad funcional mediante distancias relativas (Span, Puño, Oposición).
"""

import numpy as np
from typing import Dict, List

def _dist3d(lm_a: dict, lm_b: dict) -> float:
    """Calcula distancia 3D usando coordenadas normalizadas o reales."""
    z_a = lm_a.get("z", 0.0) if lm_a.get("z") is not None else 0.0
    z_b = lm_b.get("z", 0.0) if lm_b.get("z") is not None else 0.0
    
    return np.sqrt((lm_a["x"] - lm_b["x"])**2 + 
                   (lm_a["y"] - lm_b["y"])**2 + 
                   (z_a - z_b)**2)

def calculate_hand_metrics(landmarks: List[Dict], handedness: str = "Right") -> Dict[str, Dict]:
    """
    Calcula métricas globales de movilidad espacial de la mano.
    """
    results = {}
    
    # Unidad de Referencia (UR): Distancia de la muñeca (0) a la base del dedo medio (9)
    # Hace que las métricas sean agnósticas al tamaño de la mano del paciente y distancia de la cámara.
    ur_dist = _dist3d(landmarks[0], landmarks[9])
    
    if ur_dist < 0.001:
        return results # No se puede calcular, evitar división por cero
        
    # 1. Envergadura (Span) - Pulgar (4) a Meñique (20)
    span_dist = _dist3d(landmarks[4], landmarks[20])
    span_ratio = span_dist / ur_dist
    
    # 2. Cierre de Puño (Fist Closure) - Promedio de distancia de las yemas a la muñeca
    fist_dists = [
        _dist3d(landmarks[8], landmarks[0]),
        _dist3d(landmarks[12], landmarks[0]),
        _dist3d(landmarks[16], landmarks[0]),
        _dist3d(landmarks[20], landmarks[0])
    ]
    fist_avg = sum(fist_dists) / len(fist_dists)
    fist_ratio = fist_avg / ur_dist
    
    # 3. Oposición del Pulgar - Pulgar (4) a base de meñique (17)
    opp_dist = _dist3d(landmarks[4], landmarks[17])
    opp_ratio = opp_dist / ur_dist
    
    # 4. Separación Interdigital - Promedio entre yemas (8-12, 12-16, 16-20)
    inter_dists = [
        _dist3d(landmarks[8], landmarks[12]),
        _dist3d(landmarks[12], landmarks[16]),
        _dist3d(landmarks[16], landmarks[20])
    ]
    inter_avg = sum(inter_dists) / len(inter_dists)
    inter_ratio = inter_avg / ur_dist
    
    # Empaquetar resultados
    # Usamos el key "angle" (y "grados" en UI) conceptualmente como un Score/Ratio x100
    # para no quebrar la compatibilidad de datos del JSON de la sesión.
    
    results["mano_envergadura"] = {
        "name": "Envergadura Max",
        "angle": round(span_ratio * 100, 1),
        "status": "normal" if span_ratio > 1.3 else "limitado",
        "visibility": round(min(landmarks[4]["visibility"], landmarks[20]["visibility"]), 2),
        "draw_line": (4, 20),
        "color": (255, 0, 255) # Magenta
    }
    
    results["mano_puno"] = {
        "name": "Cierre de Puño",
        "angle": round(fist_ratio * 100, 1), # Más cercano a 0 es mejor cierre
        "status": "normal" if fist_ratio < 0.8 else "limitado",
        "visibility": round(landmarks[0]["visibility"], 2),
        "draw_line": (12, 0), # Línea representativa del dedo medio a muñeca
        "color": (0, 255, 255) # Amarillo
    }
    
    results["mano_oposicion"] = {
        "name": "Oposición Pulgar",
        "angle": round(opp_ratio * 100, 1),
        "status": "normal",
        "visibility": round(min(landmarks[4]["visibility"], landmarks[17]["visibility"]), 2),
        "draw_line": (4, 17),
        "color": (255, 165, 0) # Naranja
    }
    
    results["mano_interdigital"] = {
        "name": "Sep. Dedos",
        "angle": round(inter_ratio * 100, 1),
        "status": "normal",
        "visibility": round(landmarks[12]["visibility"], 2),
        "draw_line": (8, 12), # Representativo (Indice a Medio)
        "color": (0, 255, 0) # Verde
    }
    
    return results
