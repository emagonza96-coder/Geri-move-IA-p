from typing import List, Dict, Any

class QualityEvaluator:
    """Evalúa la calidad de la captura por frame y por sesión."""

    def __init__(self, visibility_threshold: float = 0.5):
        self.vis_threshold = visibility_threshold

    def evaluate_frame(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evalúa un solo frame.
        Retorna métricas como cantidad de puntos visibles, si el torso está visible, etc.
        """
        if not landmarks:
            return {"valid": False, "visible_count": 0, "torso_visible": False, "reason": "No_Person"}

        visible_count = sum(1 for lm in landmarks if lm.get("visibility", 0) > self.vis_threshold)
        
        # Índices de torso en MediaPipe (11, 12, 23, 24)
        torso_indices = [11, 12, 23, 24]
        torso_visible = True
        try:
            torso_visible = all(landmarks[i].get("visibility", 0) > self.vis_threshold for i in torso_indices)
        except IndexError:
            torso_visible = False

        is_valid = visible_count > 10 and torso_visible # Regla básica: necesita ver torso y al menos 10 puntos en total

        return {
            "valid": is_valid,
            "visible_count": visible_count,
            "torso_visible": torso_visible,
            "reason": "OK" if is_valid else ("Occlusion" if torso_visible else "No_Torso")
        }

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
