from typing import Any

class EventLogger:
    """Maneja el registro estructurado de eventos de la sesión."""
    def __init__(self, session_manager):
        self.session = session_manager
        
    def log(self, timestamp_ms: float, event_type: str, context: Any = ""):
        # Aquí se pueden formatear o normalizar los eventos antes de enviarlos a session
        context_str = str(context)
        self.session.add_event(timestamp_ms, event_type, context_str)
        print(f"[EVENTO] {event_type} - {context_str}")
