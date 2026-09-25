from collections import deque

class QualityTracker:
    """Calcula la calidad general de la captura usando una ventana deslizante de frames."""
    def __init__(self, window_size: int = 30):
        self.history = deque(maxlen=window_size)
        
    def add_evaluation(self, is_valid: bool):
        self.history.append(1.0 if is_valid else 0.0)
        
    def get_quality_percentage(self) -> float:
        if not self.history:
            return 0.0
        return (sum(self.history) / len(self.history)) * 100.0
