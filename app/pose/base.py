from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Dict, Any

class PoseEstimator(ABC):
    """
    Interfaz abstracta para estimadores de pose corporal.
    Permite intercambiar MediaPipe por otros modelos (MoveNet, YOLO, etc.) en el futuro.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp_ms: float) -> Optional[List[Dict[str, Any]]]:
        """
        Detecta la pose en un frame BGR.
        
        Args:
            frame: Imagen BGR.
            timestamp_ms: Tiempo en milisegundos desde el inicio de la sesión.
            
        Returns:
            Lista de landmarks detectados. Cada landmark debe ser un diccionario con al menos:
            - 'x', 'y' (coordenadas normalizadas 0-1)
            - 'visibility' (0-1)
            - Opcionalmente 'z' normalizado, o 'world_x', 'world_y', 'world_z' en metros reales.
            Retorna None si no se detecta a nadie.
        """
        pass
        
    @abstractmethod
    def release(self):
        """Libera los recursos del modelo."""
        pass
