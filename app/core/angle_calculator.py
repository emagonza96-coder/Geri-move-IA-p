"""
Motor de cálculo de ángulos articulares.
Usa trigonometría con coordenadas de landmarks de MediaPipe.
"""

import numpy as np
from typing import Dict, List


# Rangos de movilidad normal (ROM) por articulación y movimiento
NORMAL_ROM = {
    "hombro_flexion": (0, 180),
    "hombro_extension": (0, 60),
    "hombro_abduccion": (0, 180),
    "codo_flexion": (0, 145),
    "cadera_flexion": (0, 125),
    "cadera_extension": (0, 30),
    "rodilla_flexion": (0, 140),
    "tobillo_dorsiflexion": (0, 20),
    "tobillo_plantiflexion": (0, 50),
}


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calcula el ángulo en el punto b (vértice) formado por los puntos a-b-c.
    Usa coordenadas 2D [x, y] de los landmarks normalizados de MediaPipe.

    Args:
        a: Coordenadas del primer punto
        b: Coordenadas del vértice (donde se mide el ángulo)
        c: Coordenadas del tercer punto

    Returns:
        Ángulo en grados (0-180)
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
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
}


def calculate_all_angles(
    landmarks: list,
) -> Dict[str, Dict]:
    """
    Calcula todos los ángulos articulares definidos a partir de landmarks de MediaPipe.

    Args:
        landmarks: Lista de landmarks de MediaPipe (cada uno con x, y, z, visibility)

    Returns:
        Diccionario con ángulos calculados, estado, y metadata por articulación
    """
    results = {}

    for joint_key, joint_info in JOINT_ANGLES.items():
        idx_a, idx_b, idx_c = joint_info["landmarks"]

        lm_a = landmarks[idx_a]
        lm_b = landmarks[idx_b]
        lm_c = landmarks[idx_c]

        # Verificar visibilidad mínima
        min_visibility = min(lm_a["visibility"], lm_b["visibility"], lm_c["visibility"])

        if min_visibility < 0.5:
            results[joint_key] = {
                "name": joint_info["name"],
                "angle": None,
                "status": "no_visible",
                "visibility": round(min_visibility, 2),
            }
            continue

        point_a = [lm_a["x"], lm_a["y"]]
        point_b = [lm_b["x"], lm_b["y"]]
        point_c = [lm_c["x"], lm_c["y"]]

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

    return results
