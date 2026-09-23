import cv2

class KeyboardController:
    """
    Abstrae las pulsaciones de teclado y las convierte en acciones semánticas.
    """
    def __init__(self):
        pass

    def process_keys(self):
        """
        Lee el teclado y devuelve un diccionario de intenciones.
        """
        key = cv2.waitKey(1) & 0xFF
        
        actions = {
            "exit": False,
            "toggle_record": False,
            "force_record": False,
            "toggle_pause": False,
            "toggle_panel": False,
            "switch_mode": False,
            "reset_calibration": False,
            "screenshot": False,
        }

        if key == 27: # ESC
            actions["exit"] = True
        elif key == ord("q"):
            actions["toggle_record"] = True
        elif key == ord("o"):
            actions["force_record"] = True
        elif key == ord("p"):
            actions["toggle_pause"] = True
        elif key == ord("h"):
            actions["toggle_panel"] = True
        elif key == ord("m"):
            actions["switch_mode"] = True
        elif key == ord("r"):
            actions["reset_calibration"] = True
        elif key == ord("s"):
            actions["screenshot"] = True
            
        return actions
