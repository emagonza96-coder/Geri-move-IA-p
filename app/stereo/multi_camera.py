import cv2
import threading
import time
from typing import List, Tuple, Optional, Any
import numpy as np

class MultiCameraManager:
    def __init__(self, sources: List[Any]):
        """
        Inicializa la captura concurrente de múltiples cámaras.
        
        Args:
            sources: Lista de fuentes (ej. [0, 1] o ["url1", "url2"])
        """
        self.sources = sources
        self.caps = []
        self.frames = [None] * len(sources)
        self.ret_status = [False] * len(sources)
        self.running = False
        self.threads = []
        self.lock = threading.Lock()
        
        # Iniciar capturas
        for i, src in enumerate(sources):
            cap = cv2.VideoCapture(src)
            # Para evitar buffers de retraso en cámaras USB (solo funciona en algunos backends)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir la cámara {src}")
            self.caps.append(cap)
            
    def _read_thread(self, cam_idx: int):
        """Bucle de hilo independiente para leer fotogramas de una cámara específica."""
        while self.running:
            ret, frame = self.caps[cam_idx].read()
            with self.lock:
                self.ret_status[cam_idx] = ret
                if ret:
                    self.frames[cam_idx] = frame.copy()
            # Pequeño sleep para no saturar CPU (30 FPS max en el hilo lector)
            time.sleep(0.005)
            
    def start(self):
        """Inicia los hilos de lectura para todas las cámaras."""
        self.running = True
        for i in range(len(self.sources)):
            t = threading.Thread(target=self._read_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
            
    def read(self) -> Tuple[bool, List[Optional[np.ndarray]]]:
        """
        Devuelve el fotograma más reciente de cada cámara de forma sincronizada.
        Returns:
            (Exito, lista de frames)
        """
        with self.lock:
            # Comprobar si al menos tenemos frames válidos en las que pudieron abrirse
            success = any(self.ret_status)
            current_frames = list(self.frames)
            
        return success, current_frames
        
    def release(self):
        """Detiene los hilos y libera los recursos de hardware."""
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
            
        for cap in self.caps:
            cap.release()
