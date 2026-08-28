"""
EZdrive Forward Inattention Monitoring (Standalone)

  External Camera (CSI) YOLO Tracking -> Obstacle angle/level
  Internal Camera (USB) MediaPipe -> Driver head angle
  -> Determines forward inattention (state 1)

This is a standalone version, not yet merged with drowsiness detection (main.py).

Run: python3 inattention_main.py
Exit: 'q' key or Ctrl+C

Sign Calibration Mode:
  python3 inattention_main.py --calib-yaw
  Follow the on-screen instructions to turn your head left and right to get the HEAD_YAW_SIGN value.
"""

import argparse
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from picamera2 import Picamera2

from face_monitor import MODEL_PATH, estimate_head_pose
import inattention_engine as ie
from inattention_engine import InattentionDetector

# ===== External Camera / YOLO =====
YOLO_MODEL = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
YOLO_HZ = 5.0            # YOLO inference rate (considering Pi 5 load). Head detection is per frame.

# ===== Internal Camera =====
INNER_CAM_INDEX = 0

SHOW_PREVIEW = True


class HeadTracker:
    """Extracts only the head yaw from the internal camera."""

    def __init__(self, cam_index=INNER_CAM_INDEX):
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        self.cap = cv2.VideoCapture(cam_index)

    def read_yaw(self, now):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(now * 1000))
        if not result.face_landmarks:
            return None, frame
        yaw, _ = estimate_head_pose(result.face_landmarks[0], w, h)
        return yaw, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()


def run_yaw_calibration(head):
    """Determines HEAD_YAW_SIGN by making the user turn their head left and right."""
    print("\nStarting head direction sign calibration.")
    steps = [("Right", +1), ("Left", -1)]
    samples = []

    for label, direction in steps:
        print(f"\n>>> Turn your head to the {label} and hold for 3 seconds.")
        for c in (3, 2, 1):
            print(f"    {c}...")
            time.sleep(1.0)
        end = time.time() + 3.0
        vals = []
        while time.time() < end:
            yaw, _ = head.read_yaw(time.time())
            if yaw is not None:
                vals.append(yaw)
        if vals:
            med = float(np.median(vals))
            samples.append((med, direction))
            print(f"    {label} average yaw = {med:+.1f} deg")
        else:
            print("    Face not found.")

    sign = ie.calibrate_yaw_sign(samples)
    print(f"\nResult: HEAD_YAW_SIGN = {sign:+.0f}")
    print("Replace the HEAD_YAW_SIGN value in inattention_engine.py with this value.")
    return sign


def draw_overlay(frame, detections, state, info, fps):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        ang = ie.bbox_to_angle(x1, x2)
        area = ie.bbox_area_ratio(x1, y1, x2, y2)
        lv = ie.risk_level(d["class_name"], area)
        color = {0: (150, 150, 150), 1: (0, 200, 255), 2: (0, 0, 255)}[lv]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"#{d.get('track_id','-')} {d['class_name']} {ang:+.0f}deg L{lv}"
        cv2.putText(frame, tag, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Peripheral vision boundary lines
    for sign in (-1, 1):
        a = np.radians(sign * ie.PERIPHERAL_ANGLE_DEG)
        focal = 1.0 / np.tan(np.radians(ie.CAMERA_HFOV_DEG / 2.0))
        nx = np.tan(a) * focal
        x = int(ie.FRAME_WIDTH / 2 * (1 + nx))
        cv2.line(frame, (x, 0), (x, ie.FRAME_HEIGHT), (80, 80, 80), 1)

    head = info.get("head_yaw_norm")
    lines = [
        (f"STATE: {'INATTENTION(1)' if state == 1 else 'OK(0)'}",
         (0, 0, 255) if state == 1 else (0, 255, 0)),
        (f"head: {head:+.1f}deg" if head is not None else "head: N/A", (255, 255, 0)),
        (f"tracks: {info['n_tracks']}  pending: {len(info['pending'])}", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (0, 255, 255)),
    ]
    for i, (t, c) in enumerate(lines):
        cv2.putText(frame, t, (10, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    for j, p in enumerate(info["pending"]):
        cv2.putText(frame,
                    f"  #{p['id']} {p['class']} {p['angle']:+.0f}deg L{p['level']} {p['wait_s']:.1f}s",
                    (10, 140 + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-yaw", action="store_true",
                    help="Measure head yaw sign only and exit")
    args = ap.parse_args()

    head = HeadTracker()
    if not head.cap.isOpened():
        print("Could not open the internal camera.")
        return

    if args.calib_yaw:
        try:
            run_yaw_calibration(head)
        finally:
            head.close()
        return

    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    print("Starting external camera (CSI)...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (ie.FRAME_WIDTH, ie.FRAME_HEIGHT), "format": "RGB888"}))
    picam2.start()
    time.sleep(1.0)

    detector = InattentionDetector()
    detections = []
    state, info = 0, {"head_yaw_norm": None, "n_tracks": 0, "pending": [], "offenders": []}
    last_yolo = 0.0
    prev_t = time.time()

    print(f"Monitoring started (HEAD_YAW_SIGN={ie.HEAD_YAW_SIGN:+.0f}). Press 'q' to exit.")
    try:
        while True:
            now = time.time()
            frame = picam2.capture_array()

            # ---- YOLO tracking only at fixed intervals ----
            if now - last_yolo >= 1.0 / YOLO_HZ:
                last_yolo = now
                results = model.track(frame, persist=True, verbose=False)[0]
                detections = []
                if results.boxes is not None:
                    for b in results.boxes:
                        name = model.names[int(b.cls[0])]
                        if name not in TARGET_CLASSES:
                            continue
                        if float(b.conf[0]) < CONFIDENCE_THRESHOLD:
                            continue
                        tid = int(b.id[0]) if b.id is not None else None
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        detections.append({"track_id": tid, "class_name": name,
                                           "bbox": (x1, y1, x2, y2)})

            # ---- Head angle per frame ----
            yaw, _inner = head.read_yaw(now)
            state, info = detector.update(detections, yaw, now=now)

            if state == 1:
                for o in info["offenders"]:
                    print(f"[Forward Inattention] #{o['id']} {o['class']} "
                          f"{o['angle']:+.0f} deg (level {o['level']})")

            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            if SHOW_PREVIEW:
                disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_overlay(disp, detections, state, info, fps)
                cv2.imshow("EZdrive - Forward Attention", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        picam2.stop()
        head.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()