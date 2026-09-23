import cv2
import numpy as np
import time
from app.capture.video import ThreadedCapture, FileCapture
import os

def create_dummy_video(path, frames=10, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (100, 100))
    for i in range(frames):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path

def test_file_capture():
    vid_path = "test_dummy.mp4"
    create_dummy_video(vid_path, frames=5, fps=10)
    
    raw_cap = cv2.VideoCapture(vid_path)
    cap = FileCapture(raw_cap)
    
    ret, frame, ts1 = cap.read()
    assert ret is True
    assert ts1 == 0.0
    
    ret, frame, ts2 = cap.read()
    assert ret is True
    # At 10fps, frame 1 should be at 100.0 ms
    assert abs(ts2 - 100.0) < 1.0
    
    cap.release()
    if os.path.exists(vid_path):
        os.remove(vid_path)

def test_threaded_capture():
    vid_path = "test_dummy_thread.mp4"
    create_dummy_video(vid_path, frames=5, fps=30)
    
    raw_cap = cv2.VideoCapture(vid_path)
    cap = ThreadedCapture(raw_cap)
    
    time.sleep(0.1) # allow thread to read
    ret, frame, ts1 = cap.read()
    assert ret is True
    assert ts1 >= 0.0
    
    cap.release()
    if os.path.exists(vid_path):
        os.remove(vid_path)
