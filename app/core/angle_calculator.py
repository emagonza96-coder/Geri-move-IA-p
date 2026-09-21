"""
Motor de cálculo de ángulos articulares.
Usa trigonometría con coordenadas de landmarks de MediaPipe.
"""

import numpy as np
from typing import Dict, List


# Rangos de movilidad normal (ROM) por articulación y movimiento
NORMAL_ROM = {
    # Extremidades superiores
    "hombro_flexion": (0, 180),
    "hombro_extension": (0, 60),
    "hombro_abduccion": (0, 180),
    "codo_flexion": (0, 145),
    "muneca_flexion_extension": (0, 70),
    # Extremidades inferiores
    "cadera_flexion": (0, 125),
    "cadera_extension": (0, 30),
    "rodilla_flexion": (0, 140),
    "tobillo_dorsiflexion": (0, 20),
    "tobillo_plantiflexion": (0, 50),
    # Cuello (columna cervical) — valores AMA Guides
    "cuello_inclinacion": (0, 45),   # Flexion lateral (inclinarse hacia cada hombro)
    "cuello_rotacion": (0, 80),      # Rotacion (girar hacia cada lado)
    "cuello_flexion": (0, 50),       # Flexion (bajar la barbilla) - mejor en vista lateral
    # Tronco (columna lumbar + torácica)
    "tronco_inclinacion": (0, 35),   # Inclinacion lateral
    "tronco_rotacion": (0, 45),      # Rotacion del tronco
}


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calcula el ángulo en el punto b (vértice) formado por los puntos a-b-c.
    Si los puntos tienen Z (len == 3), usa álgebra 3D real (producto punto).
    Si no, hace fallback a trigonometría 2D.
    """
    if len(a) >= 3 and len(b) >= 3 and len(c) >= 3 and all(v is not None for v in [a[2], b[2], c[2]]):
        vec1 = np.array(a[:3]) - np.array(b[:3])
        vec2 = np.array(c[:3]) - np.array(b[:3])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
            
        cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
    else:
        a_2d = np.array(a[:2])
        b_2d = np.array(b[:2])
        c_2d = np.array(c[:2])

        radians = np.arctan2(c_2d[1] - b_2d[1], c_2d[0] - b_2d[0]) - np.arctan2(
            a_2d[1] - b_2d[1], a_2d[0] - b_2d[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
            
    return round(angle, 1)


def get_angle_status(angle: float, joint_key: str) -> str:
    """
    Determina si un ángulo está dentro del rango normal.

    Returns:
        'normal', 'limitado', o 'excedido'
    """
    if joint_key not in NORMAL_ROM:
        return "sin_referencia"

    min_rom, max_rom = NORMAL_ROM[joint_key]
    if min_rom <= angle <= max_rom:
        return "normal"
    elif angle < min_rom:
        return "limitado"
    else:
        return "excedido"


def _segment_tilt_from_vertical(pt_top: dict, pt_bottom: dict) -> float:
    """
    Calcula el ángulo de inclinación de un segmento respecto a la vertical.
    0° = perfectamente vertical. Aumenta cuando el segmento se inclina a cualquier lado.
    Usa coordenadas normalizadas de MediaPipe (y aumenta hacia abajo).
    """
    dx = pt_top["x"] - pt_bottom["x"]
    dy = pt_top["y"] - pt_bottom["y"]   # negativo = pt_top está arriba en imagen
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return round(angle, 1)


def _dist2d(lm_a: dict, lm_b: dict) -> float:
    """Distancia euclidiana 2D entre dos landmarks normalizados."""
    return np.sqrt((lm_a["x"] - lm_b["x"])**2 + (lm_a["y"] - lm_b["y"])**2)


def _midpoint(lm_a: dict, lm_b: dict) -> dict:
    """Punto medio entre dos landmarks, conservando Z si existe."""
    mid = {"x": (lm_a["x"] + lm_b["x"]) / 2, "y": (lm_a["y"] + lm_b["y"]) / 2}
    if "z" in lm_a and "z" in lm_b and lm_a["z"] is not None and lm_b["z"] is not None:
        mid["z"] = (lm_a["z"] + lm_b["z"]) / 2
    return mid


def _calculate_spine_angles(landmarks: list) -> Dict[str, Dict]:
    """
    Calcula ángulos de cuello y tronco usando puntos medios y análisis de segmentos.
    Diseñado para vista frontal del paciente.

    Métricas generadas:
    - cuello_inclinacion: Inclinación lateral de la cabeza (vista frontal)
    - cuello_rotacion:    Rotación de la cabeza estimada por asimetría de orejas (vista frontal)
    - tronco_inclinacion: Inclinación lateral del tronco (vista frontal)
    - tronco_rotacion:    Rotación del tronco por asimetría ancho hombros vs caderas (vista frontal)
    """
    results = {}

    nose  = landmarks[0]
    ear_l = landmarks[7]
    ear_r = landmarks[8]
    sh_l  = landmarks[11]
    sh_r  = landmarks[12]
    hip_l = landmarks[23]
    hip_r = landmarks[24]

    ears_ok    = ear_l["visibility"] > 0.5 and ear_r["visibility"] > 0.5
    shoulders_ok = sh_l["visibility"] > 0.5 and sh_r["visibility"] > 0.5
    hips_ok    = hip_l["visibility"] > 0.5 and hip_r["visibility"] > 0.5

    # 1. INCLINACIÓN LATERAL DEL CUELLO
    #    Compara el eje cabeza (midpoint orejas) vs eje tronco (midpoint hombros)
    #    Si la cabeza está centrada: ángulo ≈ 0°; si inclinada a un lado: sube
    if ears_ok and shoulders_ok:
        mid_ear = _midpoint(ear_l, ear_r)
        mid_sh  = _midpoint(sh_l, sh_r)
        tilt = _segment_tilt_from_vertical(mid_ear, mid_sh)
        results["cuello_inclinacion"] = {
            "name": "Cuello Inclinación",
            "angle": tilt,
            "status": get_angle_status(tilt, "cuello_inclinacion"),
            "visibility": round(min(ear_l["visibility"], ear_r["visibility"]), 2),
            "rom_key": "cuello_inclinacion",
            "rom_range": NORMAL_ROM["cuello_inclinacion"],
        }

    # 2. ROTACIÓN DEL CUELLO
    #    Asimetría de la distancia nariz→oreja_izq vs nariz→oreja_der
    #    Cuando la cabeza gira, la oreja del lado que se aleja parece más lejos de la nariz
    if ears_ok and nose["visibility"] > 0.5:
        d_l = _dist2d(nose, ear_l)
        d_r = _dist2d(nose, ear_r)
        total = d_l + d_r
        if total > 0.001:
            ratio = abs(d_r - d_l) / total   # 0 = recto, ~1 = totalmente girado
            rotation = round(ratio * 80.0, 1)  # escalar a grados (max ~80°)
            results["cuello_rotacion"] = {
                "name": "Cuello Rotación",
                "angle": rotation,
                "status": get_angle_status(rotation, "cuello_rotacion"),
                "visibility": round(min(nose["visibility"], ear_l["visibility"], ear_r["visibility"]), 2),
                "rom_key": "cuello_rotacion",
                "rom_range": NORMAL_ROM["cuello_rotacion"],
            }

    # 3. INCLINACIÓN LATERAL DEL TRONCO
    #    Ángulo del eje tronco (midpoint hombros → midpoint caderas) vs vertical
    #    0° = de pie derecho, aumenta cuando se inclina lateralmente
    if shoulders_ok and hips_ok:
        mid_sh  = _midpoint(sh_l, sh_r)
        mid_hip = _midpoint(hip_l, hip_r)
        trunk_tilt = _segment_tilt_from_vertical(mid_sh, mid_hip)
        results["tronco_inclinacion"] = {
            "name": "Tronco Inclinación",
            "angle": trunk_tilt,
            "status": get_angle_status(trunk_tilt, "tronco_inclinacion"),
            "visibility": round(min(sh_l["visibility"], sh_r["visibility"],
                                    hip_l["visibility"], hip_r["visibility"]), 2),
            "rom_key": "tronco_inclinacion",
            "rom_range": NORMAL_ROM["tronco_inclinacion"],
        }

    # 4. ROTACIÓN DEL TRONCO
    #    Asimetría del ancho aparente de hombros vs caderas
    #    Cuando el tronco gira, los hombros parecen más estrechos que las caderas (o viceversa)
    if shoulders_ok and hips_ok:
        shoulder_width = abs(sh_r["x"] - sh_l["x"])
        hip_width      = abs(hip_r["x"] - hip_l["x"])
        if hip_width > 0.001:
            ratio = shoulder_width / hip_width
            # ratio ≈1.0 = sin rotación; <1.0 = tronco girado
            deviation = abs(1.0 - ratio)
            rotation_est = round(min(deviation * 90.0, 45.0), 1)  # cap en 45°
            results["tronco_rotacion"] = {
                "name": "Tronco Rotación",
                "angle": rotation_est,
                "status": get_angle_status(rotation_est, "tronco_rotacion"),
                "visibility": round(min(sh_l["visibility"], sh_r["visibility"],
                                        hip_l["visibility"], hip_r["visibility"]), 2),
                "rom_key": "tronco_rotacion",
                "rom_range": NORMAL_ROM["tronco_rotacion"],
            }

    return results


# Definición de qué landmarks forman cada ángulo articular
# Formato: (landmark_a, landmark_b_vertice, landmark_c)
JOINT_ANGLES = {
    # Lado izquierdo
    "hombro_izq": {
        "landmarks": (13, 11, 23),  # codo_izq → hombro_izq → cadera_izq
        "name": "Hombro Izq",
        "rom_key": "hombro_flexion",
    },
    "codo_izq": {
        "landmarks": (11, 13, 15),  # hombro_izq → codo_izq → muñeca_izq
        "name": "Codo Izq",
        "rom_key": "codo_flexion",
    },
    "cadera_izq": {
        "landmarks": (11, 23, 25),  # hombro_izq → cadera_izq → rodilla_izq
        "name": "Cadera Izq",
        "rom_key": "cadera_flexion",
    },
    "rodilla_izq": {
        "landmarks": (23, 25, 27),  # cadera_izq → rodilla_izq → tobillo_izq
        "name": "Rodilla Izq",
        "rom_key": "rodilla_flexion",
    },
    # Lado derecho
    "hombro_der": {
        "landmarks": (14, 12, 24),  # codo_der → hombro_der → cadera_der
        "name": "Hombro Der",
        "rom_key": "hombro_flexion",
    },
    "codo_der": {
        "landmarks": (12, 14, 16),  # hombro_der → codo_der → muñeca_der
        "name": "Codo Der",
        "rom_key": "codo_flexion",
    },
    "cadera_der": {
        "landmarks": (12, 24, 26),  # hombro_der → cadera_der → rodilla_der
        "name": "Cadera Der",
        "rom_key": "cadera_flexion",
    },
    "rodilla_der": {
        "landmarks": (24, 26, 28),  # cadera_der → rodilla_der → tobillo_der
        "name": "Rodilla Der",
        "rom_key": "rodilla_flexion",
    },
    "muneca_izq": {
        "landmarks": (13, 15, 19),  # codo_izq → muneca_izq → indice_izq
        "name": "Muñeca Izq",
        "rom_key": "muneca_flexion_extension",
    },
    "muneca_der": {
        "landmarks": (14, 16, 20),  # codo_der → muneca_der → indice_der
        "name": "Muñeca Der",
        "rom_key": "muneca_flexion_extension",
    },
}


def calculate_all_angles(
    landmarks: list,
) -> Dict[str, Dict]:
    """
    Calcula todos los ángulos articulares: extremidades + cuello + tronco.

    Args:
        landmarks: Lista de landmarks de MediaPipe (cada uno con x, y, z, visibility)

    Returns:
        Diccionario con ángulos calculados, estado, y metadata por articulación
    """
    results = {}

    # --- Ángulos de extremidades (lógica existente) ---
    for joint_key, joint_info in JOINT_ANGLES.items():
        idx_a, idx_b, idx_c = joint_info["landmarks"]

        lm_a = landmarks[idx_a]
        lm_b = landmarks[idx_b]
        lm_c = landmarks[idx_c]

        min_visibility = min(lm_a["visibility"], lm_b["visibility"], lm_c["visibility"])

        if min_visibility < 0.5:
            results[joint_key] = {
                "name": joint_info["name"],
                "angle": None,
                "status": "no_visible",
                "visibility": round(min_visibility, 2),
            }
            continue

        point_a = [lm_a["x"], lm_a["y"]] + ([lm_a["z"]] if lm_a.get("z") is not None else [])
        point_b = [lm_b["x"], lm_b["y"]] + ([lm_b["z"]] if lm_b.get("z") is not None else [])
        point_c = [lm_c["x"], lm_c["y"]] + ([lm_c["z"]] if lm_c.get("z") is not None else [])

        angle = calculate_angle(point_a, point_b, point_c)
        status = get_angle_status(angle, joint_info["rom_key"])

        results[joint_key] = {
            "name": joint_info["name"],
            "angle": angle,
            "status": status,
            "visibility": round(min_visibility, 2),
            "rom_key": joint_info["rom_key"],
            "rom_range": NORMAL_ROM.get(joint_info["rom_key"]),
        }

    # --- Ángulos de cuello y tronco (cálculos especiales con puntos medios) ---
    results.update(_calculate_spine_angles(landmarks))

    return results
