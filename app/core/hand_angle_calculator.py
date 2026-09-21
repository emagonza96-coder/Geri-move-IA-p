"""
Motor de cálculo de ángulos articulares de la mano.
Mide flexión/extensión de cada articulación de cada dedo (MCP, PIP, DIP)
usando los 21 landmarks del HandLandmarker de MediaPipe.
"""

import numpy as np
from typing import Dict, List


# Rangos de movilidad normal (ROM) para articulaciones de la mano (en grados)
HAND_NORMAL_ROM = {
    # Pulgar
    "pulgar_mcp": (0, 80),
    "pulgar_ip":  (0, 80),
    # Índice
    "indice_mcp": (0, 90),
    "indice_pip": (0, 100),
    "indice_dip": (0, 80),
    # Medio
    "medio_mcp":  (0, 90),
    "medio_pip":  (0, 100),
    "medio_dip":  (0, 80),
    # Anular
    "anular_mcp": (0, 90),
    "anular_pip": (0, 100),
    "anular_dip": (0, 80),
    # Meñique
    "menique_mcp": (0, 90),
    "menique_pip": (0, 100),
    "menique_dip": (0, 80),
    # Muñeca (ángulo entre antebrazo virtual y mano)
    "muneca":      (0, 80),
}


def _calculate_angle_2d(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calcula el ángulo en el punto b (vértice) formado por a-b-c en 2D.
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


def _get_hand_angle_status(angle: float, rom_key: str) -> str:
    """Determina si un ángulo de la mano está dentro del rango normal."""
    if rom_key not in HAND_NORMAL_ROM:
        return "sin_referencia"
    min_rom, max_rom = HAND_NORMAL_ROM[rom_key]
    if min_rom <= angle <= max_rom:
        return "normal"
    elif angle < min_rom:
        return "limitado"
    else:
        return "excedido"


# Definición de articulaciones de la mano
# Formato: (landmark_a, landmark_b_vértice, landmark_c, nombre_display, rom_key)
HAND_JOINT_ANGLES = {
    # Pulgar
    "pulgar_mcp": {
        "landmarks": (1, 2, 3),   # CMC → MCP → IP
        "name": "Pulgar MCP",
        "rom_key": "pulgar_mcp",
        "vertex_idx": 2,
    },
    "pulgar_ip": {
        "landmarks": (2, 3, 4),   # MCP → IP → TIP
        "name": "Pulgar IP",
        "rom_key": "pulgar_ip",
        "vertex_idx": 3,
    },
    # Índice
    "indice_mcp": {
        "landmarks": (0, 5, 6),   # WRIST → MCP → PIP
        "name": "Índice MCP",
        "rom_key": "indice_mcp",
        "vertex_idx": 5,
    },
    "indice_pip": {
        "landmarks": (5, 6, 7),   # MCP → PIP → DIP
        "name": "Índice PIP",
        "rom_key": "indice_pip",
        "vertex_idx": 6,
    },
    "indice_dip": {
        "landmarks": (6, 7, 8),   # PIP → DIP → TIP
        "name": "Índice DIP",
        "rom_key": "indice_dip",
        "vertex_idx": 7,
    },
    # Medio
    "medio_mcp": {
        "landmarks": (0, 9, 10),  # WRIST → MCP → PIP
        "name": "Medio MCP",
        "rom_key": "medio_mcp",
        "vertex_idx": 9,
    },
    "medio_pip": {
        "landmarks": (9, 10, 11),
        "name": "Medio PIP",
        "rom_key": "medio_pip",
        "vertex_idx": 10,
    },
    "medio_dip": {
        "landmarks": (10, 11, 12),
        "name": "Medio DIP",
        "rom_key": "medio_dip",
        "vertex_idx": 11,
    },
    # Anular
    "anular_mcp": {
        "landmarks": (0, 13, 14),
        "name": "Anular MCP",
        "rom_key": "anular_mcp",
        "vertex_idx": 13,
    },
    "anular_pip": {
        "landmarks": (13, 14, 15),
        "name": "Anular PIP",
        "rom_key": "anular_pip",
        "vertex_idx": 14,
    },
    "anular_dip": {
        "landmarks": (14, 15, 16),
        "name": "Anular DIP",
        "rom_key": "anular_dip",
        "vertex_idx": 15,
    },
    # Meñique
    "menique_mcp": {
        "landmarks": (0, 17, 18),
        "name": "Meñique MCP",
        "rom_key": "menique_mcp",
        "vertex_idx": 17,
    },
    "menique_pip": {
        "landmarks": (17, 18, 19),
        "name": "Meñique PIP",
        "rom_key": "menique_pip",
        "vertex_idx": 18,
    },
    "menique_dip": {
        "landmarks": (18, 19, 20),
        "name": "Meñique DIP",
        "rom_key": "menique_dip",
        "vertex_idx": 19,
    },
}


def calculate_hand_angles(
    hand_landmarks: List[Dict],
    handedness: str = "Right",
) -> Dict[str, Dict]:
    """
    Calcula todos los ángulos articulares de una mano.

    Args:
        hand_landmarks: Lista de 21 landmarks de la mano
        handedness: "Left" o "Right" para el sufijo del nombre

    Returns:
        Diccionario de ángulos con la misma estructura que calculate_all_angles()
    """
    results = {}
    suffix = "_der" if handedness == "Right" else "_izq"

    for joint_key, joint_info in HAND_JOINT_ANGLES.items():
        idx_a, idx_b, idx_c = joint_info["landmarks"]
        full_key = joint_key + suffix

        lm_a = hand_landmarks[idx_a]
        lm_b = hand_landmarks[idx_b]
        lm_c = hand_landmarks[idx_c]

        point_a = [lm_a["x"], lm_a["y"]]
        point_b = [lm_b["x"], lm_b["y"]]
        point_c = [lm_c["x"], lm_c["y"]]

        angle = _calculate_angle_2d(point_a, point_b, point_c)
        status = _get_hand_angle_status(angle, joint_info["rom_key"])

        results[full_key] = {
            "name": joint_info["name"],
            "angle": angle,
            "status": status,
            "visibility": 1.0,
            "rom_key": joint_info["rom_key"],
            "rom_range": HAND_NORMAL_ROM.get(joint_info["rom_key"]),
            "vertex_idx": joint_info["vertex_idx"],
        }

    return results
