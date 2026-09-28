"""
Motor de cálculo de ángulos articulares.
Usa trigonometría con coordenadas de landmarks de MediaPipe.

CAMBIOS respecto a la versión original (ver comentarios "FIX:"):
  1. Conversión ángulo-crudo -> ángulo-clínico por articulación, antes de
     comparar contra expected_range y antes de exponerlo hacia afuera.
  2. Uso real de `confidence_min` (antes solo se usaba visibility_min).
  3. Suavizado temporal (EMA) del ángulo ya convertido, con estado por
     articulación, ANTES de clasificar semáforo y ANTES de guardar/mostrar.
  4. Se quita el tope duro (cap) en tronco_rotacion: un cap oculta qué tan
     mal está la fórmula en vez de dejarlo visible para poder corregirla.
  5. cuello_inclinacion / cuello_rotacion / tronco_inclinacion /
     tronco_rotacion / muneca_flexion_extension deben existir como llaves
     en settings.yaml o siempre van a caer en 'sin_referencia'. Ver nota al
     final del archivo.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from app.config.loader import load_config, get_rom, get_clinical_details


# ---------------------------------------------------------------------------
# Validación de landmarks
# ---------------------------------------------------------------------------

def _is_valid_landmark(lm: dict, min_vis: float, min_conf: float) -> bool:
    """
    FIX 2: ahora se exige también `confidence`/`presence`, no solo
    `visibility`. `visibility` en MediaPipe indica si el punto está
    presente/no-ocluido; `confidence` (o `presence`, según cómo lo esté
    nombrando tu wrapper de app.pose) indica qué tan seguro está el modelo
    de la posición (x, y, z) reportada. Un punto puede tener buena
    visibility pero mala confidence si el modelo está "adivinando" la
    coordenada por estar cerca del borde o parcialmente fuera de cuadro.

    AJUSTA el nombre de la llave `"confidence"` abajo al que realmente use
    tu diccionario de landmark (puede ser "presence", "confidence", etc.
    revisa qué devuelve app.pose para cada landmark).
    """
    if lm.get("visibility", 0) < min_vis:
        return False

    confidence = lm.get("confidence", lm.get("presence", 1.0))
    if confidence < min_conf:
        return False

    x, y = lm.get("x", -1), lm.get("y", -1)
    # Margen estricto (2% interno) para descartar miembros cortados por la cámara
    if not (0.02 <= x <= 0.98 and 0.02 <= y <= 0.98):
        return False
    return True


# ---------------------------------------------------------------------------
# Cálculo geométrico crudo (sin cambios respecto al original)
# ---------------------------------------------------------------------------

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calcula el ángulo GEOMÉTRICO CRUDO en el punto b (vértice) formado por
    los puntos a-b-c. 180° = los tres puntos casi en línea recta.
    Esto NO es todavía el ángulo clínico (ver `to_clinical_angle`).
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


# ---------------------------------------------------------------------------
# FIX 1: conversión ángulo-crudo -> ángulo-clínico, por articulación
# ---------------------------------------------------------------------------

# Para cada joint_key: True si el ángulo crudo debe invertirse (180 - raw)
# para obtener el ángulo clínico de flexión (0° = extendido).
#
# Regla usada para decidirlo (ver el mensaje anterior para el razonamiento
# geométrico completo):
#   - codo_*, cadera_*, rodilla_*: el vértice queda "recto" (colineal) en
#     posición anatómica neutra -> raw=180 cuando clínico=0 -> SÍ invertir.
#   - hombro_*: el vértice usa (codo, hombro, cadera); en posición neutra
#     (brazo pegado al cuerpo) las dos referencias ya apuntan hacia el
#     mismo lado -> raw=0 cuando clínico=0 -> NO invertir.
#
# IMPORTANTE: esto asume la convención geométrica de TU implementación
# actual de JOINT_ANGLES. Antes de confiar en esta tabla, valídala con 2-3
# posturas de referencia conocidas (ver Prioridad 1 del plan: de pie con
# transportador/foto de control) para cada joint. No la des por buena solo
# porque "tiene sentido" en el papel.
NEEDS_INVERSION = {
    "hombro_izq": False,
    "hombro_der": False,
    "codo_izq": True,
    "codo_der": True,
    "cadera_izq": True,
    "cadera_der": True,
    "rodilla_izq": True,
    "rodilla_der": True,
    "muneca_izq": True,
    "muneca_der": True,
}


def to_clinical_angle(raw_angle: float, joint_key: str) -> float:
    """Convierte el ángulo geométrico crudo al ángulo clínico (0°=extendido)."""
    if NEEDS_INVERSION.get(joint_key, False):
        return round(180.0 - raw_angle, 1)
    return raw_angle


def get_angle_status(clinical_angle: float, joint_key: str) -> str:
    """
    Determina si un ángulo CLÍNICO (ya convertido) está dentro del rango normal.
    FIX: antes recibía el ángulo crudo; ahora debe recibir siempre el
    ángulo ya pasado por `to_clinical_angle`.
    """
    min_rom, max_rom = get_rom(joint_key)
    if min_rom == 0 and max_rom == 0:
        return "sin_referencia"

    if min_rom <= clinical_angle <= max_rom:
        return "dentro_de_referencia"
    else:
        return "fuera_de_referencia"


# ---------------------------------------------------------------------------
# FIX 3: suavizado temporal (EMA) con estado por articulación
# ---------------------------------------------------------------------------

class AngleSmoother:
    """
    Suavizado exponencial (EMA) por articulación, con estado propio.

    Uso:
        smoother = AngleSmoother(alpha=0.3)
        ...
        # en cada frame, por cada joint con lectura válida:
        smoothed = smoother.update(joint_key, clinical_angle)
        # si el joint no tuvo lectura válida en este frame:
        smoother.mark_missing(joint_key)

    Una instancia de AngleSmoother debe vivir DURANTE UNA SESIÓN (se crea
    al iniciar grabación, se descarta/reinicia al terminarla). No debe ser
    un estado global compartido entre pacientes/sesiones distintas.
    """

    def __init__(self, alpha: float = 0.3, max_missing_frames: int = 15):
        """
        alpha: qué tanto peso tiene la lectura nueva (0-1). Más alto =
               reacciona más rápido pero suaviza menos. 0.25-0.35 es un
               buen punto de partida; ajústalo comparando el jitter
               resultante contra movimiento real conocido.
        max_missing_frames: si un joint lleva más de este número de frames
               consecutivos sin lectura válida, se descarta el valor
               suavizado (se vuelve a partir de cero) en vez de arrastrar
               indefinidamente el último valor conocido.
        """
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self._state: Dict[str, float] = {}
        self._missing_count: Dict[str, int] = {}

    def update(self, joint_key: str, clinical_angle: float) -> float:
        self._missing_count[joint_key] = 0
        prev = self._state.get(joint_key)
        if prev is None:
            smoothed = clinical_angle
        else:
            smoothed = self.alpha * clinical_angle + (1 - self.alpha) * prev
        self._state[joint_key] = smoothed
        return round(smoothed, 1)

    def mark_missing(self, joint_key: str) -> None:
        """Llamar cuando el joint no tuvo lectura válida en este frame."""
        self._missing_count[joint_key] = self._missing_count.get(joint_key, 0) + 1
        if self._missing_count[joint_key] > self.max_missing_frames:
            self._state.pop(joint_key, None)

    def reset(self) -> None:
        """Llamar al iniciar una sesión nueva / un paciente nuevo."""
        self._state.clear()
        self._missing_count.clear()


# ---------------------------------------------------------------------------
# Ángulos de cuello / tronco
# ---------------------------------------------------------------------------

def _segment_tilt_from_vertical(pt_top: dict, pt_bottom: dict) -> float:
    dx = pt_top["x"] - pt_bottom["x"]
    dy = pt_top["y"] - pt_bottom["y"]
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return round(angle, 1)


def _dist2d(lm_a: dict, lm_b: dict) -> float:
    return np.sqrt((lm_a["x"] - lm_b["x"]) ** 2 + (lm_a["y"] - lm_b["y"]) ** 2)


def _midpoint(lm_a: dict, lm_b: dict) -> dict:
    mid = {"x": (lm_a["x"] + lm_b["x"]) / 2, "y": (lm_a["y"] + lm_b["y"]) / 2}
    if "z" in lm_a and "z" in lm_b and lm_a["z"] is not None and lm_b["z"] is not None:
        mid["z"] = (lm_a["z"] + lm_b["z"]) / 2
    return mid


def _calculate_spine_angles(landmarks: list, min_vis: float, min_conf: float) -> Dict[str, Dict]:
    """
    FIX 4: se elimina el tope duro (`min(..., 45.0)`) en tronco_rotacion.
    Un cap oculta el valor real y hace imposible saber qué tan mal está la
    fórmula. Es preferible ver "rotacion_est = 187" (un valor absurdo que
    delata inmediatamente que la fórmula está mal) que ver siempre "45"
    (que parece un dato real y no lo es).

    NOTA: estas 4 métricas + muneca_flexion_extension necesitan existir
    como llaves en settings.yaml (clinical_rom) o siempre van a devolver
    'sin_referencia'. Además, la fórmula de cuello_rotacion y
    tronco_rotacion (basadas en asimetría de distancias/anchos) da
    valores de ~45° incluso con el paciente de frente en el video de
    prueba -> esto necesita validarse con posturas de referencia (cabeza
    recta = debería dar ~0°) antes de confiar en el número, no solo
    quitar el cap.
    """
    results = {}

    nose  = landmarks[0]
    ear_l = landmarks[7]
    ear_r = landmarks[8]
    sh_l  = landmarks[11]
    sh_r  = landmarks[12]
    hip_l = landmarks[23]
    hip_r = landmarks[24]

    ears_ok      = _is_valid_landmark(ear_l, min_vis, min_conf) and _is_valid_landmark(ear_r, min_vis, min_conf)
    shoulders_ok = _is_valid_landmark(sh_l, min_vis, min_conf) and _is_valid_landmark(sh_r, min_vis, min_conf)
    hips_ok      = _is_valid_landmark(hip_l, min_vis, min_conf) and _is_valid_landmark(hip_r, min_vis, min_conf)
    nose_ok      = _is_valid_landmark(nose, min_vis, min_conf)

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
            "rom_range": get_rom("cuello_inclinacion"),
        }

    if ears_ok and nose_ok:
        d_l = _dist2d(nose, ear_l)
        d_r = _dist2d(nose, ear_r)
        total = d_l + d_r
        if total > 0.001:
            ratio = abs(d_r - d_l) / total
            rotation = round(ratio * 80.0, 1)
            results["cuello_rotacion"] = {
                "name": "Rotación de Cuello",
                "angle": rotation,
                "status": get_angle_status(rotation, "cuello_rotacion"),
                "visibility": round(min(nose["visibility"], ear_l["visibility"], ear_r["visibility"]), 2),
                "rom_key": "cuello_rotacion",
                "rom_range": get_rom("cuello_rotacion"),
            }

    if shoulders_ok and hips_ok:
        mid_sh  = _midpoint(sh_l, sh_r)
        mid_hip = _midpoint(hip_l, hip_r)
        trunk_tilt = _segment_tilt_from_vertical(mid_sh, mid_hip)
        results["tronco_inclinacion"] = {
            "name": "Inclinación Lat. Tronco",
            "angle": trunk_tilt,
            "status": get_angle_status(trunk_tilt, "tronco_inclinacion"),
            "visibility": round(min(sh_l["visibility"], sh_r["visibility"],
                                    hip_l["visibility"], hip_r["visibility"]), 2),
            "rom_key": "tronco_inclinacion",
            "rom_range": get_rom("tronco_inclinacion"),
        }

    if shoulders_ok and hips_ok:
        shoulder_width = abs(sh_r["x"] - sh_l["x"])
        hip_width      = abs(hip_r["x"] - hip_l["x"])
        if hip_width > 0.001:
            ratio = shoulder_width / hip_width
            deviation = abs(1.0 - ratio)
            # FIX 4: sin cap. Ver nota arriba.
            rotation_est = round(deviation * 90.0, 1)
            results["tronco_rotacion"] = {
                "name": "Rotación de Tronco",
                "angle": rotation_est,
                "status": get_angle_status(rotation_est, "tronco_rotacion"),
                "visibility": round(min(sh_l["visibility"], sh_r["visibility"],
                                        hip_l["visibility"], hip_r["visibility"]), 2),
                "rom_key": "tronco_rotacion",
                "rom_range": get_rom("tronco_rotacion"),
            }

    return results


# ---------------------------------------------------------------------------
# Definición de articulaciones (sin cambios respecto al original)
# ---------------------------------------------------------------------------

JOINT_ANGLES = {
    "hombro_izq": {"landmarks": (13, 11, 23), "name": "Hombro Izq", "rom_key": "hombro_flexion"},
    "codo_izq":   {"landmarks": (11, 13, 15), "name": "Codo Izq",   "rom_key": "codo_flexion"},
    "cadera_izq": {"landmarks": (11, 23, 25), "name": "Cadera Izq", "rom_key": "cadera_flexion"},
    "rodilla_izq":{"landmarks": (23, 25, 27), "name": "Rodilla Izq","rom_key": "rodilla_flexion"},
    "hombro_der": {"landmarks": (14, 12, 24), "name": "Hombro Der", "rom_key": "hombro_flexion"},
    "codo_der":   {"landmarks": (12, 14, 16), "name": "Codo Der",   "rom_key": "codo_flexion"},
    "cadera_der": {"landmarks": (12, 24, 26), "name": "Cadera Der", "rom_key": "cadera_flexion"},
    "rodilla_der":{"landmarks": (24, 26, 28), "name": "Rodilla Der","rom_key": "rodilla_flexion"},
    "muneca_izq": {"landmarks": (13, 15, 19), "name": "Muñeca Flex/Ext Izq", "rom_key": "muneca_flexion_extension"},
    "muneca_der": {"landmarks": (14, 16, 20), "name": "Muñeca Flex/Ext Der", "rom_key": "muneca_flexion_extension"},
}


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def calculate_all_angles(
    landmarks: list,
    smoother: Optional[AngleSmoother] = None,
) -> Dict[str, Dict]:
    """
    Calcula todos los ángulos articulares: extremidades + cuello + tronco.

    Args:
        landmarks: Lista de landmarks de MediaPipe (x, y, z, visibility, confidence).
        smoother: instancia de AngleSmoother viva durante la sesión. Si es
                  None, no se aplica suavizado (útil para pruebas unitarias
                  o análisis frame-a-frame puntual, pero NO recomendado
                  para la sesión en vivo).

    Returns:
        Diccionario con ángulo crudo, ángulo clínico (crudo o suavizado
        según `smoother`), estado, y metadata por articulación.
    """
    results = {}

    config = load_config()
    min_vis = config.get("thresholds", {}).get("visibility_min", 0.5)
    min_conf = config.get("thresholds", {}).get("confidence_min", 0.7)  # FIX 2: ahora sí se usa

    for joint_key, joint_info in JOINT_ANGLES.items():
        idx_a, idx_b, idx_c = joint_info["landmarks"]
        lm_a, lm_b, lm_c = landmarks[idx_a], landmarks[idx_b], landmarks[idx_c]

        is_valid = (
            _is_valid_landmark(lm_a, min_vis, min_conf)
            and _is_valid_landmark(lm_b, min_vis, min_conf)
            and _is_valid_landmark(lm_c, min_vis, min_conf)
        )
        min_visibility = min(lm_a["visibility"], lm_b["visibility"], lm_c["visibility"])

        if not is_valid:
            if smoother is not None:
                smoother.mark_missing(joint_key)
            results[joint_key] = {
                "name": joint_info["name"],
                "angle": None,
                "status": "fuera_de_cuadro",
                "visibility": round(min_visibility, 2),
            }
            continue

        point_a = [lm_a["x"], lm_a["y"]] + ([lm_a["z"]] if lm_a.get("z") is not None else [])
        point_b = [lm_b["x"], lm_b["y"]] + ([lm_b["z"]] if lm_b.get("z") is not None else [])
        point_c = [lm_c["x"], lm_c["y"]] + ([lm_c["z"]] if lm_c.get("z") is not None else [])

        raw_angle = calculate_angle(point_a, point_b, point_c)
        clinical_angle = to_clinical_angle(raw_angle, joint_key)          # FIX 1

        if smoother is not None:
            clinical_angle = smoother.update(joint_key, clinical_angle)   # FIX 3

        status = get_angle_status(clinical_angle, joint_info["rom_key"])
        details = get_clinical_details(joint_info["rom_key"])

        results[joint_key] = {
            "name": joint_info["name"],
            "angle": clinical_angle,      # ángulo clínico, ya convertido y suavizado
            "angle_raw": raw_angle,       # se conserva para depuración/auditoría
            "status": status,
            "visibility": round(min_visibility, 2),
            "rom_key": joint_info["rom_key"],
            "rom_range": details.get("expected_range", (0.0, 0.0)),
            "warning": details.get("warning"),
        }

    results.update(_calculate_spine_angles(landmarks, min_vis, min_conf))

    return results