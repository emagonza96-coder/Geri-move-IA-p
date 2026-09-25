class FramingValidator:
    """
    Mantiene el estado de encuadre válido a lo largo del tiempo.
    Requiere un número sostenido de frames válidos antes de permitir iniciar
    acciones críticas como la grabación (evita falsos positivos).
    """
    def __init__(self, required_frames: int = 15):
        self.required_frames = required_frames
        self.valid_count = 0
        self.is_ready = False

    def update(self, is_valid: bool) -> bool:
        if is_valid:
            self.valid_count += 1
            if self.valid_count >= self.required_frames:
                self.is_ready = True
        else:
            self.valid_count = 0
            self.is_ready = False
        return self.is_ready
        
    def reset(self):
        self.valid_count = 0
        self.is_ready = False
