import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from collections import deque
from app.config.loader import get_framing_config, get_quality_config, get_ui_config, get_threshold

from app.pose.mediapipe_impl import MediaPipePoseEstimator
from app.pose.hand import HandDetector
from app.biomechanics.angles import calculate_all_angles
from app.biomechanics.hand import evaluate_biomechanics
from app.processing.profile import CalibrationProfile
from app.ui.video_overlay import VideoOverlay
from app.ui.panel_renderer import PanelRenderer
from app.ui.composer import DisplayComposer
from app.capture.video import ThreadedCapture, FileCapture
from app.quality.evaluator import QualityEvaluator
from app.core.session import SessionManager
from app.core.input import KeyboardController
from app.core.framing_validator import FramingValidator
from app.core.quality_tracker import QualityTracker
from app.core.event_logger import EventLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Mobility Scan")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--mode", type=str, default="body", choices=["body", "hand"])
    return parser.parse_args()

class MobilityApp:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        
        # Input setup
        if args.source.isdigit():
            self.source = int(args.source)
            source_name = f"webcam_{self.source}"
        else:
            self.source = args.source
            source_name = Path(args.source).stem
            
        _raw_cap = cv2.VideoCapture(self.source)
        if not _raw_cap.isOpened():
            print(f"[ERROR] No se pudo abrir {self.source}")
            sys.exit(1)
            
        if args.width > 0 and isinstance(self.source, int):
            _raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            _raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
            _raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_w = int(_raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(_raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = _raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(_raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.is_live = isinstance(self.source, int) or str(self.source).startswith(("http", "rtsp"))
        if self.is_live:
            self.cap = ThreadedCapture(_raw_cap)
        else:
            self.cap = FileCapture(_raw_cap)
            
        self.proc_w = args.width if args.width > 0 and args.width < frame_w else frame_w
        self.proc_h = int(frame_h * (args.width / frame_w)) if args.width > 0 and args.width < frame_w else frame_h

        # Modules
        self.session = SessionManager(self.output_dir, self.source, frame_w, frame_h, fps_source, source_name)
        self.input_handler = KeyboardController()
        
        self.detector = MediaPipePoseEstimator(min_detection_confidence=args.confidence, model_complexity=args.model)
        self.hand_detector = HandDetector(min_detection_confidence=args.confidence)
        
        framing_conf = get_framing_config()
        quality_conf = get_quality_config()
        ui_conf = get_ui_config()
        vis_min = get_threshold("visibility_min", 0.5)
        
        self.evaluator = QualityEvaluator(visibility_threshold=vis_min)
        self.profile = CalibrationProfile()
        self.framing_validator = FramingValidator(required_frames=framing_conf.get("required_consecutive_frames", 15))
        self.framing_status = {"valid": False, "reason": "Evaluando..."}
        self.quality_tracker = QualityTracker(window_size=quality_conf.get("window_size", 30))
        self.event_logger = EventLogger(self.session)
        
        self.video_overlay = VideoOverlay()
        self.panel_renderer = PanelRenderer(width=ui_conf.get("panel_width", 240), height=frame_h)
        self.composer = DisplayComposer()
        
        # State
        self.active_mode = "HAND" if args.mode == "hand" else "BODY"
        self.paused = False
        self.panel_hidden = False
        
        self.frame_count = 0
        self.fps_display = 0.0
        self.cached_ts = ""
        self.ts_update_time = 0
        
        self.framing_error_msg = ""
        self.framing_error_time = 0.0
        
        self.mouse_state = {"dragging": None, "x": 0, "y": 0, "orig_x": 0, "orig_y": 0}
        self.current_landmarks = []
        
        def _on_mouse(event, x, y, flags, param):
            if self.session.recording_active or not self.current_landmarks:
                return
            nx, ny = x / frame_w, y / frame_h
            self.mouse_state["x"] = nx
            self.mouse_state["y"] = ny
            if event == cv2.EVENT_LBUTTONDOWN:
                min_dist = float('inf')
                nearest_idx = None
                for idx, lm in enumerate(self.current_landmarks):
                    if lm.get("visibility", 0) > 0.4:
                        dist = ((lm["x"] - nx)**2 + (lm["y"] - ny)**2)**0.5
                        if dist < 0.05 and dist < min_dist:
                            min_dist = dist
                            nearest_idx = idx
                if nearest_idx is not None:
                    self.mouse_state["dragging"] = nearest_idx
                    lm_adj = self.current_landmarks[nearest_idx]
                    dx_curr, dy_curr = self.profile.offsets.get(nearest_idx, (0.0, 0.0))
                    self.mouse_state["orig_x"] = lm_adj["x"] - dx_curr
                    self.mouse_state["orig_y"] = lm_adj["y"] - dy_curr
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.mouse_state["dragging"] is not None:
                    idx = self.mouse_state["dragging"]
                    dx = nx - self.mouse_state["orig_x"]
                    dy = ny - self.mouse_state["orig_y"]
                    self.profile.set_offset(idx, dx, dy)
            elif event == cv2.EVENT_LBUTTONUP:
                self.mouse_state["dragging"] = None

        if not self.args.headless:
            cv2.namedWindow("Mobility Scan")
            cv2.setMouseCallback("Mobility Scan", _on_mouse)

    def run(self):
        prev_time = time.time()
        last_processed_ts = -1.0
        timestamp_ms = 0
        
        try:
            while self.cap.isOpened():
                # 1. User Input
                actions = self.input_handler.process_keys()
                if actions["exit"]:
                    break
                if actions["toggle_pause"]:
                    self.paused = not self.paused
                    self.event_logger.log(timestamp_ms, "pause_toggled", str(self.paused))
                if actions["toggle_panel"]:
                    self.panel_hidden = not self.panel_hidden
                    self.event_logger.log(timestamp_ms, "panel_toggled", str(self.panel_hidden))
                if actions["switch_mode"]:
                    self.active_mode = "HAND" if self.active_mode == "BODY" else "BODY"
                    self.event_logger.log(timestamp_ms, "mode_switched", self.active_mode)
                if actions["reset_calibration"] and not self.session.recording_active:
                    self.profile.reset()
                    self.event_logger.log(timestamp_ms, "calibration_reset", "")

                if self.paused:
                    continue

                # 2. Capture
                ret, frame, frame_timestamp_ms = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                if frame_timestamp_ms <= last_processed_ts:
                    time.sleep(0.005)
                    continue
                
                last_processed_ts = frame_timestamp_ms
                timestamp_ms = int(frame_timestamp_ms)
                self.frame_count += 1
                
                if self.args.max_frames > 0 and self.frame_count > self.args.max_frames:
                    break

                current_time = time.time()
                dt = current_time - prev_time
                if dt > 0:
                    self.fps_display = 0.8 * self.fps_display + 0.2 * (1.0 / dt)
                prev_time = current_time
                
                if current_time - self.ts_update_time >= 1.0:
                    self.cached_ts = time.strftime("%H:%M:%S")
                    self.ts_update_time = current_time

                # 3. Process
                proc_frame = cv2.resize(frame, (self.proc_w, self.proc_h)) if self.proc_w != frame.shape[1] else frame
                angles = {}
                detected = False
                current_landmarks = []
                
                # Copia cruda del frame antes de que VideoOverlay lo modifique (in-place)
                raw_frame = frame.copy()

                if self.active_mode == "BODY":
                    raw_landmarks = self.detector.detect(proc_frame, timestamp_ms)
                    if raw_landmarks:
                        self.framing_status = self.evaluator.check_framing(raw_landmarks)
                        self.framing_validator.update(self.framing_status["valid"])
                        self.quality_tracker.add_evaluation(self.framing_status["valid"])
                        
                        detected = True
                        self.current_landmarks = self.profile.apply(raw_landmarks)
                        angles = calculate_all_angles(self.current_landmarks)
                        frame = self.video_overlay.draw_skeleton(frame, self.current_landmarks, angles, self.panel_hidden)
                    else:
                        self.framing_validator.update(False)
                        self.quality_tracker.add_evaluation(False)

                elif self.active_mode == "HAND":
                    hands = self.hand_detector.detect(proc_frame)
                    if hands:
                        detected = True
                        self.current_landmarks = self.profile.apply(hands[0]["landmarks"])
                        angles = evaluate_biomechanics(self.current_landmarks)
                        deformities = angles.get("deformidades", []) if isinstance(angles, dict) else []
                        frame = self.hand_detector.draw_hand(frame, hands, cam_w=frame.shape[1], deformities=deformities)

                # 4. Handle Record Toggle and Screenshots
                if actions.get("screenshot"):
                    screenshot_path = self.output_dir / f"captura_{self.session.session_id}_f{self.frame_count}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                    self.screenshot_time = time.time()
                    print(f"[INFO] Captura guardada: {screenshot_path}")

                if actions["toggle_record"]:
                    if self.session.recording_active:
                        self.session.stop_recording(timestamp_ms, self.active_mode)
                        self.framing_validator.reset()
                    else:
                        if self.active_mode == "BODY":
                            if self.framing_validator.is_ready:
                                self.session.start_recording(timestamp_ms, self.active_mode)
                                self.framing_error_msg = ""
                            else:
                                self.framing_error_msg = self.framing_status.get("reason", "Paciente no estabilizado")
                                self.framing_error_time = time.time()
                        else:
                            self.session.start_recording(timestamp_ms, self.active_mode)
                            
                if actions["force_record"] and not self.session.recording_active:
                    self.session.start_recording(timestamp_ms, self.active_mode, override=True)
                    self.framing_error_msg = ""

                # 5. Save Data (Video Crudo)
                self.session.add_frame(raw_frame, timestamp_ms, self.active_mode, angles)

                # 6. UI Render
                frame_h, frame_w = frame.shape[:2]

                session_metadata = {
                    "id": "USR-001",
                    "task": "Evaluacion",
                    "intent": 1,
                    "view": "Frontal",
                    "timer": self.session.get_elapsed_timer()
                }

                quality_indicator = self.quality_tracker.get_quality_percentage()

                if self.panel_hidden:
                    panel_surface = np.zeros((frame_h, 0, 3), dtype=np.uint8)
                else:
                    panel_surface = self.panel_renderer.render(
                        angles=angles if self.active_mode == "BODY" else {},
                        fps=self.fps_display,
                        frame_num=self.frame_count,
                        cached_ts=self.cached_ts,
                        active_mode=self.active_mode,
                        hand_metrics=angles if self.active_mode == "HAND" else None,
                        session_info=session_metadata,
                        quality_indicator=quality_indicator,
                        is_recording=self.session.recording_active
                    )

                # Componer
                frame = self.composer.compose(frame, panel_surface)

                if self.framing_error_msg and (time.time() - self.framing_error_time < 3.0):
                    cv2.rectangle(frame, (frame_w - 300, frame_h - 40), (frame_w - 10, frame_h - 10), (0, 0, 150), -1)
                    cv2.putText(frame, self.framing_error_msg, (frame_w - 290, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if getattr(self, 'screenshot_time', 0.0) > 0 and (time.time() - self.screenshot_time < 1.0):
                    cv2.putText(frame, "CAPTURA GUARDADA", (frame_w // 2 - 110, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                if not self.session.recording_active:
                    for idx, lm in enumerate(self.current_landmarks):
                        if lm.get("_adjusted", False):
                            px, py = int(lm["x"] * frame_w), int(lm["y"] * frame_h)
                            cv2.circle(frame, (px, py), 6, (0, 215, 255), -1)
                            cv2.circle(frame, (px, py), 8, (0, 100, 200), 1)

                if not self.args.headless:
                    cv2.imshow("Mobility Scan", frame)

        except KeyboardInterrupt:
            print("\n[INFO] Interrumpido por el usuario")

    def cleanup(self):
        self.session.close()
        self.cap.release()
        if self.detector: self.detector.release()
        if self.hand_detector: self.hand_detector.release()
        if not self.args.headless:
            cv2.destroyAllWindows()

def main():
    args = parse_args()
    app = MobilityApp(args)
    app.run()
    app.cleanup()

if __name__ == "__main__":
    main()
