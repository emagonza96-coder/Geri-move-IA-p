import cv2
import csv
from pathlib import Path

class FrameRecorder:
    """
    Gestiona la grabación de videos simultáneos (crudo, esqueleto y pantalla completa) 
    y el registro de timestamps por frame en un archivo CSV.
    """
    def __init__(self, base_video_path: str, csv_path: str, fps: float, width: int, height: int, screen_width: int):
        self.base_path = Path(base_video_path)
        self.csv_path = Path(csv_path)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        name = self.base_path.stem
        dir_path = self.base_path.parent
        
        self.writer_raw = cv2.VideoWriter(
            str(dir_path / f"{name}_raw.mp4"),
            fourcc, fps, (width, height)
        )
        self.writer_skeleton = cv2.VideoWriter(
            str(dir_path / f"{name}_skeleton.mp4"),
            fourcc, fps, (width, height)
        )
        self.writer_screen = cv2.VideoWriter(
            str(dir_path / f"{name}_screen.mp4"),
            fourcc, fps, (screen_width, height)
        )
        
        # Preparar archivo CSV para sincronización
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_idx", "timestamp_ms"])
        
        self.recorded_frames = 0
        
    def add_frames(self, raw_frame, skeleton_frame, screen_frame, timestamp_ms: float):
        if self.writer_raw:
            self.writer_raw.write(raw_frame)
            self.writer_skeleton.write(skeleton_frame)
            self.writer_screen.write(screen_frame)
            
            self.csv_writer.writerow([self.recorded_frames, timestamp_ms])
            self.recorded_frames += 1

    def close(self):
        if self.writer_raw: self.writer_raw.release()
        if self.writer_skeleton: self.writer_skeleton.release()
        if self.writer_screen: self.writer_screen.release()
        if self.csv_file: self.csv_file.close()

    def get_recorded_frames(self):
        return self.recorded_frames
