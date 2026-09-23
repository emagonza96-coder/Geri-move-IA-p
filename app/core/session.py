import cv2
import json
import time
import os
from pathlib import Path
from datetime import datetime

class SessionManager:
    """
    Gestiona la inicialización de archivos, el log de datos estructurados (JSON), 
    la grabación en video crudo (MP4) y los eventos de la sesión.
    """
    def __init__(self, output_dir, source, frame_w, frame_h, fps_source, source_name="webcam_0"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"sesion_{self.session_id}_{source_name}.mp4"
        self.json_path = self.output_dir / f"sesion_{self.session_id}_{source_name}_data.json"
        
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps_source = fps_source
        
        # Inicializar JSON Structure
        self.session_data = {
            "metadata": {
                "source": str(source),
                "date": datetime.now().isoformat(),
                "fps_source": fps_source,
                "resolution": f"{frame_w}x{frame_h}"
            },
            "events": [],
            "frames": []
        }
        
        # VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            fps_source,
            (frame_w, frame_h),
        )
        
        self.recording_active = False
        self.record_start_time = 0.0
        self.recorded_frames = 0
        
        print(f"[INFO] SessionManager inicializado.")
        print(f"[INFO] Destino MP4: {self.video_path}")
        print(f"[INFO] Destino JSON: {self.json_path}")
        
    def start_recording(self, timestamp_ms, mode, override=False):
        if not self.recording_active:
            self.recording_active = True
            self.record_start_time = time.time()
            event_type = "start_recording_override" if override else "start_recording"
            self.add_event(timestamp_ms, event_type, mode)
            print(f"[INFO] GRABANDO SECCIÓN ({mode.upper()})")
            
    def stop_recording(self, timestamp_ms, mode):
        if self.recording_active:
            self.recording_active = False
            self.add_event(timestamp_ms, "stop_recording", mode)
            print(f"[INFO] Grabación pausada/detenida. Frames grabados: {self.recorded_frames}")

    def add_event(self, timestamp_ms, event_type, context=""):
        self.session_data["events"].append({
            "timestamp_ms": timestamp_ms,
            "type": event_type,
            "context": context
        })

    def add_frame(self, frame_img, timestamp_ms, mode, angles_dict):
        """Añade el frame de video al MP4 y los datos JSON al log, SOLO si se está grabando."""
        if not self.recording_active:
            return False

        # 1. Guardar la imagen cruda
        if self.writer:
            self.writer.write(frame_img)
            
        # 2. Guardar métricas
        angles_clean = {}
        for k, v in angles_dict.items():
            angles_clean[k] = {
                "angle": v.get("angle"), 
                "status": v.get("status"), 
                "visibility": v.get("visibility")
            }
            
        self.session_data["frames"].append({
            "frame_idx": self.recorded_frames,
            "timestamp_ms": timestamp_ms,
            "mode": mode.lower(),
            "angles": angles_clean
        })
        
        self.recorded_frames += 1
        return True

    def get_elapsed_timer(self):
        """Devuelve un string (MM:SS) si está grabando, o '00:00' si no."""
        if self.recording_active:
            elapsed = int(time.time() - self.record_start_time)
            return f"{elapsed//60:02d}:{elapsed%60:02d}"
        return "00:00"

    def generate_summary(self):
        """Genera un resumen estadístico de todas las mediciones de la sesión."""
        if not self.session_data["frames"]:
            return {}

        import numpy as np
        from app.biomechanics.angles import JOINT_ANGLES
        
        angle_data = {}
        for m in self.session_data["frames"]:
            for joint_key, data in m.get("angles", {}).items():
                if data.get("angle") is not None:
                    if joint_key not in angle_data:
                        angle_data[joint_key] = []
                    angle_data[joint_key].append(data["angle"])

        summary = {}
        for joint_key, values in angle_data.items():
            if values:
                summary[joint_key] = {
                    "name": JOINT_ANGLES.get(joint_key, {}).get("name", joint_key),
                    "min": round(min(values), 1),
                    "max": round(max(values), 1),
                    "avg": round(sum(values) / len(values), 1),
                    "std": round(float(np.std(values)), 1),
                    "samples": len(values),
                }
        return summary

    def close(self):
        """Cierra el writer y guarda el JSON. Borra basura si no se grabó nada."""
        if self.writer:
            self.writer.release()

        self.session_data["metadata"]["total_frames_processed"] = self.recorded_frames
        self.session_data["summary"] = self.generate_summary()

        # Guardar JSON
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

        print("\n==================================================")
        print("  SESIÓN COMPLETADA")
        print("==================================================")
        print(f"  Frames grabados: {self.recorded_frames}")
        
        if self.recorded_frames == 0:
            print("  [INFO] No se guardaron archivos (0 frames grabados).")
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            if os.path.exists(self.json_path):
                os.remove(self.json_path)
        else:
            print(f"  Video: {self.video_path.name}")
            print(f"  Datos: {self.json_path.name}")
            
            # Mostrar resumen de ángulos
            summary = self.session_data["summary"]
            if summary:
                print(f"\n  RESUMEN DE ÁNGULOS (promedios)")
                print(f"  {'─'*40}")
                for joint_key, stats in summary.items():
                    print(
                        f"  {stats['name']:16s}  "
                        f"min={stats['min']:5.1f}°  "
                        f"max={stats['max']:5.1f}°  "
                        f"avg={stats['avg']:5.1f}°"
                    )
                print()
        print("==================================================")
