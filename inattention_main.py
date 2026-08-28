"""
EZdrive forward-attention monitor (standalone).

  External camera (CSI) + YOLO tracking -> obstacle bearing and risk level
  Internal camera (USB) + MediaPipe     -> driver head yaw
  -> forward-attention state (0 ok / 1 inattention)
  -> alert_output.AlertSystem.update_output() for the buzzer/LED/vibration warning

This module only ever emits state 0 or 1 through AlertSystem. Drowsiness
(state 2/3) is handled separately in main.py; when the two are combined,
call update_output(state=max(drowsy_state, inattention_state), ...) instead.

The power button (tact 1) on AlertSystem arms/disarms the buzzer output as
usual; this script keeps running and detecting regardless, so the preview
window still shows live numbers even before the system is armed.

Run:   python3 inattention_main.py
Quit:  q key (closes the preview), or Ctrl+C

Sign calibration (run this once before first use):
       python3 inattention_main.py --calib-yaw
Turn your head left and right when prompted; the script reports the
HEAD_YAW_SIGN value to put into inattention_engine.py.
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
from alert_output import AlertSystem

# ===== External camera / YOLO =====
YOLO_MODEL = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
YOLO_HZ = 5.0            # detection rate; head checks still run every frame

# How long an inattention event (state 1) is held after it fires, so the
# buzzer pattern in alert_output.py (short beep, long gap) has time to sound.
INATTENTION_HOLD_SEC = 2.0

# ===== Internal camera =====
INNER_CAM_INDEX = 0
INNER_CAM_WIDTH = 320     # lower resolution -> much faster MediaPipe inference
INNER_CAM_HEIGHT = 240
HEAD_YAW_SKIP = 1         # run MediaPipe every (SKIP+1)-th frame; reuse last value otherwise
                          # 0 = every frame, 1 = every other frame, etc.

SHOW_PREVIEW = True


class HeadTracker:
    """Reads head yaw from the internal camera."""

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, INNER_CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INNER_CAM_HEIGHT)
        # Keep the capture buffer at 1 frame so read() always returns the
        # latest frame instead of a stale one queued up behind it.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._skip_counter = 0
        self._last_yaw = None

    def read_yaw(self, now):
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        # Skip MediaPipe inference on some frames and reuse the last yaw,
        # since head motion is slow compared to the camera frame rate.
        if HEAD_YAW_SKIP > 0:
            self._skip_counter = (self._skip_counter + 1) % (HEAD_YAW_SKIP + 1)
            if self._skip_counter != 0:
                return self._last_yaw, frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(now * 1000))
        if not result.face_landmarks:
            self._last_yaw = None
            return None, frame
        yaw, _ = estimate_head_pose(result.face_landmarks[0], w, h)
        self._last_yaw = yaw
        return yaw, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()


def run_yaw_calibration(head):
    """Ask the driver to turn left and right, then report HEAD_YAW_SIGN."""
    print("\nHead-yaw sign calibration.")
    steps = [("RIGHT", +1), ("LEFT", -1)]
    samples = []

    for label, direction in steps:
        print(f"\n>>> Turn your head {label} and hold for 3 seconds.")
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
            print(f"    {label}: median yaw = {med:+.1f} deg")
        else:
            print("    No face detected.")

    sign = ie.calibrate_yaw_sign(samples)
    print(f"\nResult: HEAD_YAW_SIGN = {sign:+.0f}")
    print("Set HEAD_YAW_SIGN in inattention_engine.py to this value.")
    return sign


def draw_overlay(frame, detections, state, info, fps):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        ang = ie.bbox_to_angle(x1, x2)
        area = ie.bbox_area_ratio(x1, y1, x2, y2)
        lv = ie.risk_level(d["class_name"], area)
        color = {0: (150, 150, 150), 1: (0, 200, 255), 2: (0, 0, 255)}[lv]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"#{d.get('track_id', '-')} {d['class_name']} {ang:+.0f}deg L{lv}"
        cv2.putText(frame, tag, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Peripheral-vision boundary lines
    focal = 1.0 / np.tan(np.radians(ie.CAMERA_HFOV_DEG / 2.0))
    for sign in (-1, 1):
        nx = np.tan(np.radians(sign * ie.PERIPHERAL_ANGLE_DEG)) * focal
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
                    f"  #{p['id']} {p['class']} {p['angle']:+.0f}deg "
                    f"L{p['level']} {p['wait_s']:.1f}s",
                    (10, 140 + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-yaw", action="store_true",
                    help="measure the head-yaw sign and exit")
    args = ap.parse_args()

    head = HeadTracker()
    if not head.cap.isOpened():
        print("Cannot open the internal camera.")
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

    alert = AlertSystem()
    detector = InattentionDetector()
    detections = []
    state = 0
    info = {"head_yaw_norm": None, "n_tracks": 0, "pending": [], "offenders": []}
    last_yolo = 0.0
    prev_t = time.time()
    warn_until = 0.0     # keep state=1 (and the buzzer) alive for a short hold time

    print(f"Monitoring (HEAD_YAW_SIGN={ie.HEAD_YAW_SIGN:+.0f}).")
    print("Press the power button (tact 1) to arm the alert output. Press q to quit the preview.")
    try:
        while True:
            now = time.time()
            frame = picam2.capture_array()

            # ---- run YOLO tracking at a fixed rate ----
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

            # ---- head pose every frame ----
            yaw, _inner = head.read_yaw(now)
            raw_state, info = detector.update(detections, yaw, now=now)

            # An inattention event fires for a single instant; hold it briefly so the
            # buzzer pattern (BUZZER_PATTERNS[1]) has time to actually sound.
            if raw_state == 1:
                warn_until = now + INATTENTION_HOLD_SEC
            state = 1 if now < warn_until else 0

            if raw_state == 1:
                for o in info["offenders"]:
                    print(f"[INATTENTION] #{o['id']} {o['class']} "
                          f"{o['angle']:+.0f}deg (level {o['level']})")

            # This module only ever reports state 0/1. Drowsiness (state 2/3) is
            # handled by main.py; when combined later, pass max(drowsy_state, state) here.
            alert.update_output(state=state, ir_reliable=True)

            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            if SHOW_PREVIEW:
                disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_overlay(disp, detections, state, info, fps)
                cv2.imshow("EZdrive - Forward Attention", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        picam2.stop()
        head.close()
        alert.cleanup()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
