"""
Motor de Biomecánica Clínica de Mano (Tiempo Real).
Evaluación goniométrica, deformidades articulares estáticas y abducción.
"""

import numpy as np
from typing import Dict, List

def _dist3d(lm_a: dict, lm_b: dict) -> float:
    z_a = lm_a.get("z", 0.0) if lm_a.get("z") is not None else 0.0
    z_b = lm_b.get("z", 0.0) if lm_b.get("z") is not None else 0.0
    return np.sqrt((lm_a["x"] - lm_b["x"])**2 + (lm_a["y"] - lm_b["y"])**2 + (z_a - z_b)**2)

def _vector(lm_start: dict, lm_end: dict) -> np.ndarray:
    z_s = lm_start.get("z", 0.0) if lm_start.get("z") is not None else 0.0
    z_e = lm_end.get("z", 0.0) if lm_end.get("z") is not None else 0.0
    return np.array([lm_end["x"] - lm_start["x"], lm_end["y"] - lm_start["y"], z_e - z_s])

def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def _calculate_goniometry(a: dict, b: dict, c: dict) -> float:
    """
    Calcula el grado clínico de flexión.
    0° = completamente recto (180° interno geométrico).
    90° = flexionado en L.
    < 0° = Hiperextensión.
    """
    v1 = _vector(b, a) # Vector de vértice hacia atrás
    v2 = _vector(b, c) # Vector de vértice hacia adelante
    
    internal_angle = _angle_between_vectors(v1, v2)
    # Goniometría clínica: 180 (recto) geométrico -> 0 flexión clínica
    flexion = 180.0 - internal_angle
    return round(flexion, 1)

def evaluate_biomechanics(landmarks: List[Dict]) -> Dict[str, Dict]:
    """
    Realiza la evaluación estática (Frame-a-Frame) de la biomecánica de la mano.
    """
    results = {}
    
    # 1. Goniometría Clínica (MCF, IFP, IFD)
    # Definición de articulaciones
    joints = {
        "pulgar_mcf": (1, 2, 3), "pulgar_if": (2, 3, 4),
        "indice_mcf": (0, 5, 6), "indice_ifp": (5, 6, 7), "indice_ifd": (6, 7, 8),
        "medio_mcf":  (0, 9, 10), "medio_ifp": (9, 10, 11), "medio_ifd": (10, 11, 12),
        "anular_mcf": (0, 13, 14), "anular_ifp": (13, 14, 15), "anular_ifd": (14, 15, 16),
        "menique_mcf": (0, 17, 18), "menique_ifp": (17, 18, 19), "menique_ifd": (18, 19, 20)
    }
    
    angles_data = {}
    for j_name, (idx_a, idx_b, idx_c) in joints.items():
        if all(landmarks[i]["visibility"] > 0.5 for i in (idx_a, idx_b, idx_c)):
            flex = _calculate_goniometry(landmarks[idx_a], landmarks[idx_b], landmarks[idx_c])
            angles_data[j_name] = flex
        else:
            angles_data[j_name] = None

    results["goniometria"] = angles_data
    
    # 2. Deformidades
    deformities = []
    fingers = ["indice", "medio", "anular", "menique"]
    for finger in fingers:
        ifp = angles_data.get(f"{finger}_ifp")
        ifd = angles_data.get(f"{finger}_ifd")
        
        if ifp is not None and ifd is not None:
            # Cuello de cisne: IFP en hiperextensión (< -5°), IFD en flexión (> 20°)
            if ifp < -5 and ifd > 20:
                deformities.append(f"{finger.capitalize()}: Cuello de Cisne")
            
            # Ojal (Boutonniere): IFP muy flexionado (> 40°), IFD en hiperextensión (< -5°)
            if ifp > 40 and ifd < -5:
                deformities.append(f"{finger.capitalize()}: Ojal (Boutonniere)")
                
    results["deformidades"] = deformities
    
    # 3. Abducción del Pulgar (Útil para rizartrosis)
    # Ángulo entre vector del pulgar (2 -> 4) y vector del índice (5 -> 8)
    if all(landmarks[i]["visibility"] > 0.5 for i in (2, 4, 5, 8)):
        v_thumb = _vector(landmarks[2], landmarks[4])
        v_index = _vector(landmarks[5], landmarks[8])
        abd_angle = _angle_between_vectors(v_thumb, v_index)
        results["abduccion_pulgar"] = round(abd_angle, 1)
    else:
        results["abduccion_pulgar"] = None
        
    # 4. Índice de Cierre de Puño (Normalizado)
    ur_dist = _dist3d(landmarks[0], landmarks[9]) # Tamaño mano (Muñeca a MCF Medio)
    if ur_dist > 0.001:
        fist_dists = [
            _dist3d(landmarks[8], landmarks[0]),
            _dist3d(landmarks[12], landmarks[0]),
            _dist3d(landmarks[16], landmarks[0]),
            _dist3d(landmarks[20], landmarks[0])
        ]
        fist_avg = sum(fist_dists) / 4.0
        # 0 = Abierta, ~1 = Puño cerrado. Normalización inversa para lógica.
        # Si la distancia promedio a la muñeca es casi el tamaño de la mano, está abierta (fist_avg / ur_dist ≈ 1.0)
        # Si está cerrado, fist_avg / ur_dist ≈ 0.2
        # Para cumplir "0 = abierta, 1 = cerrado", invertimos:
        closure_ratio = max(0.0, 1.2 - (fist_avg / ur_dist)) 
        results["indice_cierre"] = min(1.0, closure_ratio) # Acotar a [0, 1]
    else:
        results["indice_cierre"] = 0.0

    # 5. Oposición tipo Kapandji (Escala simplificada de 10 niveles)
    # Niveles 1-10 según distancia normalizada de la punta del pulgar (4) a zonas clave de cada dedo
    # Ref: Kapandji AI. Clinical test of apposition and counter-apposition of the thumb.
    KAPANDJI_TARGETS = [
        # (nivel, idx_landmark_objetivo, descripcion)
        (1,  6,  "IFP Indice"),     # pulgar toca nudillo IFP índice
        (2,  7,  "IFD Indice"),     # pulgar toca nudillo IFD índice
        (3, 10,  "IFP Medio"),
        (4, 11,  "IFD Medio"),
        (5, 14,  "IFP Anular"),
        (6, 15,  "IFD Anular"),
        (7, 18,  "IFP Menique"),
        (8, 19,  "IFD Menique"),
        (9, 20,  "Punta Menique"),
        (10, 17, "Base Menique MCF"),
    ]
    
    ur_dist2 = _dist3d(landmarks[0], landmarks[9])
    kapandji_nivel = 0
    
    if ur_dist2 > 0.001:
        CONTACT_THRESHOLD = 0.35  # Fracción del tamaño de la mano para "contacto"
        thumb_tip = landmarks[4]
        
        for nivel, target_idx, _ in KAPANDJI_TARGETS:
            dist = _dist3d(thumb_tip, landmarks[target_idx])
            normalized = dist / ur_dist2
            if normalized < CONTACT_THRESHOLD:
                kapandji_nivel = nivel  # Tomamos el nivel más alto alcanzado
                
    results["kapandji_nivel"] = kapandji_nivel
    
    return results
