"""
Perfil de calibración por paciente.
Almacena offsets de posición por landmark para corregir errores sistemáticos
del modelo de MediaPipe para anatomías específicas.

Flujo:
  1. MediaPipe detecta landmarks (posiciones brutas)
  2. CalibrationProfile.apply() suma los offsets guardados
  3. El resto del sistema (ángulos, HUD) usa las posiciones corregidas
"""

import json
import os
from typing import Dict, List, Optional, Tuple


class CalibrationProfile:
    """
    Perfil de calibración en memoria que almacena offsets (dx, dy) por índice de landmark.
    Los offsets se expresan en coordenadas normalizadas MediaPipe [0.0, 1.0].
    Solo existe durante la sesión activa.
    """

    def __init__(self):
        # Dict[landmark_idx -> (dx, dy)]  en coordenadas normalizadas
        self.offsets: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Aplicación de offsets
    # ------------------------------------------------------------------

    def apply(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Aplica los offsets almacenados a una lista de landmarks.
        Solo modifica landmarks que tengan un offset definido y visibilidad > 0.4.
        Retorna una NUEVA lista (no modifica la original).
        """
        if not self.offsets:
            return landmarks

        corrected = []
        for idx, lm in enumerate(landmarks):
            if idx in self.offsets and lm.get("visibility", 1.0) > 0.4:
                dx, dy = self.offsets[idx]
                corrected.append({
                    **lm,
                    "x": max(0.0, min(1.0, lm["x"] + dx)),
                    "y": max(0.0, min(1.0, lm["y"] + dy)),
                    "_adjusted": True,   # marcador interno para el HUD
                })
            else:
                corrected.append(lm)
        return corrected

    # ------------------------------------------------------------------
    # Manejo de offsets individuales
    # ------------------------------------------------------------------

    def set_offset(self, landmark_idx: int, dx: float, dy: float):
        """Guarda o actualiza el offset de un landmark."""
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            # Offset demasiado pequeño → limpiar
            self.offsets.pop(landmark_idx, None)
        else:
            self.offsets[landmark_idx] = (round(dx, 4), round(dy, 4))

    def clear_offset(self, landmark_idx: int):
        """Elimina el offset de un landmark específico."""
        if landmark_idx in self.offsets:
            self.offsets.pop(landmark_idx)

    def reset(self):
        """Elimina todos los offsets del perfil."""
        self.offsets.clear()

    def adjusted_indices(self) -> set:
        """Retorna el conjunto de índices de landmarks con offset activo."""
        return set(self.offsets.keys())
