from typing import Dict, List, Any

def get_real_world_coordinates(world_landmarks: List[Dict[str, Any]], height_m: float = None) -> List[Dict[str, float]]:
    """
    Toma los pose_world_landmarks de MediaPipe (estimados en metros).
    Si se provee la altura real del paciente, aplica un factor de escala.
    """
    if not world_landmarks:
        return []

    # Si no hay altura provista, usamos las estimaciones crudas en metros de MediaPipe
    if not height_m:
        return [{"x": lm.get("world_x", 0), "y": lm.get("world_y", 0), "z": lm.get("world_z", 0)} for lm in world_landmarks]

    # Calcular la altura estimada por MediaPipe
    # Aproximación: distancia de (punto medio talones) a (cabeza/nariz)
    # 0 = nariz, 29/30 = talones (aproximadamente, o 27/28 tobillos)
    try:
        nose = world_landmarks[0]
        ankle_l = world_landmarks[27]
        ankle_r = world_landmarks[28]
        
        mid_ankle_y = (ankle_l.get("world_y", 0) + ankle_r.get("world_y", 0)) / 2
        # y decrece hacia arriba en MediaPipe (cabeza es negativo respecto a caderas)
        estimated_height = abs(mid_ankle_y - nose.get("world_y", 0))
        
        scale_factor = 1.0
        if estimated_height > 0.1:
            scale_factor = height_m / estimated_height
            
        scaled_landmarks = []
        for lm in world_landmarks:
            scaled_landmarks.append({
                "x": lm.get("world_x", 0) * scale_factor,
                "y": lm.get("world_y", 0) * scale_factor,
                "z": lm.get("world_z", 0) * scale_factor,
            })
        return scaled_landmarks
        
    except IndexError:
        # Si no hay suficientes landmarks
        return [{"x": lm.get("world_x", 0), "y": lm.get("world_y", 0), "z": lm.get("world_z", 0)} for lm in world_landmarks]
