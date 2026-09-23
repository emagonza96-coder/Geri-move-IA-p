import cv2
import threading
import time
from typing import Tuple, Optional
import numpy as np

class ThreadedCapture:
    """
    Wrapper de cv2.VideoCapture que lee frames en un hilo separado.
    Siempre entrega el frame más reciente, eliminando el lag del buffer.
    Ahora devuelve (ret, frame, timestamp_ms) donde timestamp es el momento exacto de captura.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self._cap     = cap
        self._frame   = None
        self._ret     = False
        self._timestamp_ms = 0.0
        self._lock    = threading.Lock()
        self._running = True
        self._start_time = time.time()
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            capture_time = (time.time() - self._start_time) * 1000.0
            with self._lock:
                self._ret   = ret
                self._frame = frame
                self._timestamp_ms = capture_time

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Devuelve (éxito, frame, timestamp_en_ms)"""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return self._ret, self._frame.copy(), self._timestamp_ms

    def isOpened(self): return self._cap.isOpened()
    def get(self, prop): return self._cap.get(prop)
    def set(self, prop, val): return self._cap.set(prop, val)

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._cap.release()

class FileCapture:
    """
    Wrapper para archivos de video que respeta la misma interfaz que ThreadedCapture
    devolviendo (ret, frame, timestamp_ms) pero basado en los FPS del archivo.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ret, frame = self._cap.read()
        timestamp_ms = (self._frame_idx / self._fps) * 1000.0
        if ret:
            self._frame_idx += 1
        return ret, frame, timestamp_ms

    def isOpened(self): return self._cap.isOpened()
    def get(self, prop): return self._cap.get(prop)
    def set(self, prop, val): return self._cap.set(prop, val)

    def release(self):
        self._cap.release()
