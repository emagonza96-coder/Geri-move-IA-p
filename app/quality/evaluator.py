from typing import List, Dict, Any

class QualityEvaluator:
    """Evalúa la calidad de la captura por frame y por sesión."""

    def __init__(self, visibility_threshold: float = 0.5):
        self.vis_threshold = visibility_threshold

    def check_framing(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifica que los landmarks esenciales estén dentro del cuadro y visibles.
        Retorna: {"valid": bool, "reason": str}
        """
        if not landmarks:
            return {"valid": False, "reason": "Paciente no detectado"}

        # Diccionario de índices esenciales
        essential = {
            "Cabeza": [0], # Nariz
            "Hombros": [11, 12],
            "Caderas": [23, 24],
            "Rodillas": [25, 26],
            "Pies": [27, 28] # Tobillos
        }

        for part, indices in essential.items():
            for idx in indices:
                try:
                    lm = landmarks[idx]
                    if lm.get("visibility", 0) < self.vis_threshold:
                        return {"valid": False, "reason": f"{part} con baja confianza: ajuste iluminación"}
                    
                    x, y = lm.get("x", -1), lm.get("y", -1)
                    # Tolerancia estricta para encuadre
                    if not (0.02 <= x <= 0.98 and 0.02 <= y <= 0.98):
                        return {"valid": False, "reason": f"{part} fuera de cuadro: aleje la cámara"}
                except IndexError:
                    return {"valid": False, "reason": f"{part} no detectada"}

        return {"valid": True, "reason": "Encuadre OK"}

    def summarize_session(self, frame_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estadísticas generales de la sesión."""
        total = len(frame_evaluations)
        if total == 0:
            return {"valid_ratio": 0.0, "total_frames": 0}
            
        valid_frames = sum(1 for ev in frame_evaluations if ev["valid"])
        return {
            "total_frames": total,
            "valid_frames": valid_frames,
            "valid_ratio": round(valid_frames / total, 3)
        }
