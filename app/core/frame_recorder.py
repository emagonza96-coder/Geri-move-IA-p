import cv2
import csv
from pathlib import Path

class FrameRecorder:
    """
    Gestiona la grabación de video crudo y el registro de timestamps por frame en un archivo CSV.
    """
    def __init__(self, video_path: str, csv_path: str, fps: float, width: int, height: int):
        self.video_path = Path(video_path)
        self.csv_path = Path(csv_path)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            fps,
            (width, height),
        )
        
        # Preparar archivo CSV para sincronización
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_idx", "timestamp_ms"])
        
        self.recorded_frames = 0
        
    def add_frame(self, frame, timestamp_ms: float):
        if self.writer:
            self.writer.write(frame)
            self.csv_writer.writerow([self.recorded_frames, timestamp_ms])
            self.recorded_frames += 1

    def close(self):
        if self.writer:
            self.writer.release()
        if self.csv_file:
            self.csv_file.close()

    def get_recorded_frames(self):
        return self.recorded_frames
