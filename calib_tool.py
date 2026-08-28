"""
EZdrive threshold calibration tool.

Shows live numbers (no decision logic applied) so the three thresholds
can be picked by observation:

  1) PERIPHERAL_ANGLE_DEG  - obstacle bearing, printed live as objects move
  2) AREA_THRESHOLDS       - bounding-box area ratio, printed per class/distance
  3) CONFIRM_RATIO / MIN   - head yaw angle, printed live while turning

Run:   python3 calib_tool.py
Quit:  q key, or Ctrl+C

Log:   press SPACE to write the current frame's numbers to calib_log.csv,
       so you can review distances/angles afterwards.
"""

import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

from face_monitor import MODEL_PATH, estimate_head_pose
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import inattention_engine as ie

YOLO_MODEL = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
INNER_CAM_INDEX = 0
LOG_PATH = "calib_log.csv"


class HeadTracker:
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
            return None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(now * 1000))
        if not result.face_landmarks:
            return None
        yaw, _ = estimate_head_pose(result.face_landmarks[0], w, h)
        return yaw

    def close(self):
        self.cap.release()
        self.landmarker.close()


def main():
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    print("Starting external camera (CSI)...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (ie.FRAME_WIDTH, ie.FRAME_HEIGHT), "format": "RGB888"}))
    picam2.start()
    time.sleep(1.0)

    head = HeadTracker()
    if not head.cap.isOpened():
        print("Cannot open the internal camera.")
        picam2.stop()
        return

    log_rows = []
    print("\nReady. SPACE = log current numbers, q = quit.\n")

    try:
        while True:
            now = time.time()
            frame = picam2.capture_array()

            results = model.track(frame, persist=True, verbose=False)[0]
            objs = []
            if results.boxes is not None:
                for b in results.boxes:
                    name = model.names[int(b.cls[0])]
                    if name not in TARGET_CLASSES:
                        continue
                    if float(b.conf[0]) < CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    ang = ie.bbox_to_angle(x1, x2)
                    area = ie.bbox_area_ratio(x1, y1, x2, y2)
                    tid = int(b.id[0]) if b.id is not None else -1
                    objs.append({"id": tid, "class": name, "bbox": (x1, y1, x2, y2),
                                "angle": ang, "area": area})

            head_yaw = head.read_yaw(now)
            head_norm = ie.normalize_head_yaw(head_yaw)

            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for o in objs:
                x1, y1, x2, y2 = o["bbox"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp,
                           f"#{o['id']} {o['class']}  angle={o['angle']:+.1f}deg  area={o['area']:.4f}",
                           (x1, max(y1 - 8, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # centre marker + current peripheral-angle guide line (default 20 deg)
            focal = 1.0 / np.tan(np.radians(ie.CAMERA_HFOV_DEG / 2.0))
            for deg in (20, 30):
                for sign in (-1, 1):
                    nx = np.tan(np.radians(sign * deg)) * focal
                    x = int(ie.FRAME_WIDTH / 2 * (1 + nx))
                    color = (0, 200, 255) if deg == 20 else (0, 100, 150)
                    cv2.line(disp, (x, 0), (x, ie.FRAME_HEIGHT), color, 1)
                    cv2.putText(disp, f"{deg}", (x + 3, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            head_txt = f"{head_norm:+.1f}deg" if head_norm is not None else "N/A"
            lines = [
                f"HEAD YAW (normalised): {head_txt}",
                f"objects tracked: {len(objs)}",
                "SPACE = log this frame   q = quit",
            ]
            for i, t in enumerate(lines):
                cv2.putText(disp, t, (10, ie.FRAME_HEIGHT - 70 + i * 24),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            cv2.imshow("EZdrive - Threshold Calibration", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                for o in objs:
                    row = {"t": round(now, 2), "class": o["class"],
                          "angle_deg": round(o["angle"], 1),
                          "area_ratio": round(o["area"], 4),
                          "head_yaw_deg": round(head_norm, 1) if head_norm is not None else ""}
                    log_rows.append(row)
                    print(f"  logged: {row}")
                if not objs:
                    row = {"t": round(now, 2), "class": "", "angle_deg": "",
                          "area_ratio": "",
                          "head_yaw_deg": round(head_norm, 1) if head_norm is not None else ""}
                    log_rows.append(row)
                    print(f"  logged (head only): {row}")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        picam2.stop()
        head.close()
        cv2.destroyAllWindows()

        if log_rows:
            with open(LOG_PATH, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["t", "class", "angle_deg",
                                                   "area_ratio", "head_yaw_deg"])
                w.writeheader()
                w.writerows(log_rows)
            print(f"Saved {len(log_rows)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
