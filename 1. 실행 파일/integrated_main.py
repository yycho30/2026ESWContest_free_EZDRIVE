"""
EZdrive integrated monitor (Raspberry Pi 5).

Single process, single alert output:

  Internal camera (USB) + MediaPipe -> head pose, yawning
  IR sensor (Pro Mini over BLE)     -> eyelid distance
  External camera (CSI) + YOLO      -> obstacle bearing and risk level
        |
        +-> DrowsinessDetector  (RF model)      -> state 0 / 2 / 3
        +-> InattentionDetector (rule based)    -> state 0 / 1
        |
        +-> AlertSystem.update_output()         -> LED / buzzer / vibration

State priority
  3 asleep  >  2 drowsy  >  1 inattention  >  0 normal
  The higher number is the more urgent warning, so the combined state is
  simply max(drowsy_state, inattention_state).

Per-driver adaptation
  1. Startup calibration (10 s) fixes IR open/closed references and the
     driver's neutral head angles.
  2. During driving the IR open reference keeps adapting from frames judged
     "eyes open", so slow sensor drift does not break the normalisation.
  3. The first minute of driving sets a personal RF probability threshold,
     so the same model suits drivers whose baseline probabilities differ.

Run:   python3 integrated_main.py
Quit:  q key (preview window) or Ctrl+C
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from picamera2 import Picamera2

from face_monitor import (
    MODEL_PATH, MAR_THRESHOLD, YAWN_HOLD_TIME,
    calc_mar, estimate_head_pose,
)
from drowsiness_engine import CalibrationCollector, DrowsinessDetector
import inattention_engine as ie
from inattention_engine import InattentionDetector
from alert_output import AlertSystem
from shared_ble_link import SharedBLELink

# ===== Internal camera =====
# Fixed /dev/videoN indices are not reliable on this Pi: USB re-enumeration
# after a reboot or cable reconnect can swap which index the internal USB
# webcam gets (seen both at 8/9 and at 0/1). Finding it by device name is
# stable across reboots; only fall back to a fixed index if that fails.
INNER_CAM_NAME = "Innomaker-U20CAM"
INNER_CAM_INDEX_FALLBACK = 0


def find_camera_index_by_name(name_substr, max_index=40):
    """Scan /sys/class/video4linux for a capture device whose name matches,
    and return its index, or None if not found."""
    import glob
    for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            with open(path) as f:
                dev_name = f.read().strip()
        except OSError:
            continue
        if name_substr.lower() in dev_name.lower():
            idx = int(path.split("video4linux/video")[1].split("/")[0])
            if idx <= max_index:
                return idx
    return None
INNER_CAM_WIDTH = 320       # low resolution keeps MediaPipe fast
INNER_CAM_HEIGHT = 240
HEAD_YAW_SKIP = 1           # run MediaPipe every (SKIP+1)-th frame, reuse last result

# ===== External camera / YOLO =====
YOLO_MODEL = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
YOLO_HZ = 5.0

# ===== Timing =====
INFER_HZ = 10.0             # drowsiness inference rate, matches the training data
INFER_INTERVAL = 1.0 / INFER_HZ

SHOW_PREVIEW = True


def parse_ir(raw):
    """'IR value: 359' -> 359.0"""
    if raw is None:
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return float(digits) if digits else None


class FaceTracker:
    """Head pose, angular velocity and yawning from the internal camera."""

    def __init__(self, cam_index=None):
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )

        if cam_index is None:
            cam_index = find_camera_index_by_name(INNER_CAM_NAME)
            if cam_index is None:
                print(f"Could not find a camera named '{INNER_CAM_NAME}', "
                      f"falling back to index {INNER_CAM_INDEX_FALLBACK}. "
                      f"Run 'v4l2-ctl --list-devices' to check.")
                cam_index = INNER_CAM_INDEX_FALLBACK
            else:
                print(f"Internal camera '{INNER_CAM_NAME}' found at /dev/video{cam_index}")

        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, INNER_CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INNER_CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always read the newest frame

        self.prev_yaw = self.prev_pitch = self.prev_time = None
        self.yawn_count = 0
        self.is_yawning = False
        self.mouth_open_start = None
        self.mouth_open_duration = 0.0

        self._skip_counter = 0
        self._last = {
            "yaw": None, "pitch": None, "yaw_vel": 0.0, "pitch_vel": 0.0,
            "is_yawning": False, "mouth_open_duration": 0.0, "face_found": False,
        }

    def process(self, now):
        ok, frame = self.cap.read()
        if not ok:
            return dict(self._last), None

        # Reuse the previous result on skipped frames; head motion is slow
        # compared with the camera frame rate.
        if HEAD_YAW_SKIP > 0:
            self._skip_counter = (self._skip_counter + 1) % (HEAD_YAW_SKIP + 1)
            if self._skip_counter != 0:
                return dict(self._last), frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(now * 1000))

        yaw = pitch = None
        yaw_vel = pitch_vel = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            yaw, pitch = estimate_head_pose(lm, w, h)

            if yaw is not None and self.prev_yaw is not None and self.prev_time is not None:
                dt = now - self.prev_time
                if dt > 0:
                    yaw_vel = (yaw - self.prev_yaw) / dt
                    pitch_vel = (pitch - self.prev_pitch) / dt
            if yaw is not None:
                self.prev_yaw, self.prev_pitch, self.prev_time = yaw, pitch, now

            mar = calc_mar(lm, w, h)
            if mar > MAR_THRESHOLD:
                if self.mouth_open_start is None:
                    self.mouth_open_start = now
                self.mouth_open_duration = now - self.mouth_open_start
                if self.mouth_open_duration >= YAWN_HOLD_TIME and not self.is_yawning:
                    self.is_yawning = True
                    self.yawn_count += 1
            else:
                self.mouth_open_start = None
                self.mouth_open_duration = 0.0
                self.is_yawning = False

        self._last = {
            "yaw": yaw, "pitch": pitch,
            "yaw_vel": yaw_vel, "pitch_vel": pitch_vel,
            "is_yawning": self.is_yawning,
            "mouth_open_duration": self.mouth_open_duration,
            "face_found": bool(result.face_landmarks),
        }
        return dict(self._last), frame

    def close(self):
        self.cap.release()
        self.landmarker.close()


def draw_overlay(frame, mode, guide, face, ir_val,
                 drowsy_state, inatt_state, final_state,
                 ir_reliable, dinfo, iinfo, detections, fps):
    """Overlay drawn on the external camera image."""
    # obstacle boxes
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        ang = ie.bbox_to_angle(x1, x2)
        area = ie.bbox_area_ratio(x1, y1, x2, y2)
        lv = ie.risk_level(d["class_name"], area)
        color = {0: (150, 150, 150), 1: (0, 200, 255), 2: (0, 0, 255)}[lv]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{d.get('track_id','-')} {d['class_name']} {ang:+.0f}deg L{lv}",
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # peripheral boundary lines
    focal = 1.0 / np.tan(np.radians(ie.CAMERA_HFOV_DEG / 2.0))
    for sign in (-1, 1):
        nx = np.tan(np.radians(sign * ie.PERIPHERAL_ANGLE_DEG)) * focal
        x = int(ie.FRAME_WIDTH / 2 * (1 + nx))
        cv2.line(frame, (x, 0), (x, ie.FRAME_HEIGHT), (80, 80, 80), 1)

    lines = []
    if mode == "waiting":
        lines.append(("Press POWER button to start", (200, 200, 200)))
    elif mode == "calibrating":
        lines.append((f"CALIBRATION: {guide}", (0, 255, 255)))
    else:
        label = {0: "NORMAL", 1: "INATTENTION", 2: "DROWSY", 3: "ASLEEP"}[final_state]
        color = {0: (0, 255, 0), 1: (0, 200, 255),
                 2: (0, 165, 255), 3: (0, 0, 255)}[final_state]
        lines.append((f"STATE: {label}", color))
        lines.append((f"drowsy={drowsy_state}  inattention={inatt_state}", (255, 255, 255)))
        if dinfo:
            warm = "  [warmup]" if dinfo["warmup"] else ""
            lines.append((f"prob {dinfo['prob']:.2f} / th {dinfo['threshold']:.2f}{warm}",
                          (255, 255, 255)))
            lines.append((f"ir_norm {dinfo['ir_norm']:.2f}  closed {dinfo['closed_s']:.1f}s"
                          f"  perclos {dinfo['perclos']:.2f}", (255, 255, 255)))
        if not ir_reliable:
            lines.append(("IR UNRELIABLE", (0, 0, 255)))

    head = iinfo.get("head_yaw_norm") if iinfo else None
    head_txt = f"{head:+.0f}deg" if head is not None else "N/A"
    face_txt = "O" if (face and face["face_found"]) else "X"
    ir_txt = ir_val if ir_val is not None else "N/A"
    lines.append((f"IR {ir_txt}   face {face_txt}   head {head_txt}", (255, 255, 0)))
    lines.append((f"FPS {fps:.1f}", (0, 255, 255)))

    for i, (t, c) in enumerate(lines):
        cv2.putText(frame, t, (10, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    if iinfo:
        for j, p in enumerate(iinfo.get("pending", [])):
            cv2.putText(frame,
                        f"  #{p['id']} {p['class']} {p['angle']:+.0f}deg "
                        f"L{p['level']} {p['wait_s']:.1f}s",
                        (10, 30 + len(lines) * 26 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    print("Starting external camera (CSI)...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (ie.FRAME_WIDTH, ie.FRAME_HEIGHT), "format": "RGB888"}))
    picam2.start()
    time.sleep(1.0)

    face_tracker = FaceTracker()
    if not face_tracker.cap.isOpened():
        print("Cannot open the internal camera.")
        picam2.stop()
        return

    # AlertSystem opens the one and only serial connection to the master
    # HM-10 (/dev/serial0). It must be created before SharedBLELink, which
    # reads incoming IR lines on that same connection.
    alert = AlertSystem()
    ble_link = SharedBLELink(alert)

    mode = "waiting"            # waiting -> calibrating -> running
    collector = None
    drowsy_det = None
    inatt_det = None
    guide = ""

    drowsy_state = inatt_state = final_state = 0
    ir_reliable = True
    dinfo = None
    iinfo = {"head_yaw_norm": None, "n_tracks": 0, "pending": [], "offenders": []}
    detections = []

    last_yolo = 0.0
    last_infer = 0.0
    inatt_warned = False
    prev_t = time.time()

    print("Press the power button (tact 1) to start. Press q to quit.")
    try:
        while True:
            now = time.time()
            ext_frame = picam2.capture_array()
            face, _inner_frame = face_tracker.process(now)
            ir_val = parse_ir(ble_link.get_latest_ir())

            # ---- power button starts (or restarts) calibration ----
            if alert.consume_calibration_request():
                collector = CalibrationCollector()
                drowsy_det = None
                inatt_det = None
                mode = "calibrating"
                alert.begin_calibration()
                print("Calibration started. Follow the on-screen guide.")
            if not alert.system_on and mode != "waiting":
                mode = "waiting"
                alert.end_calibration()
                print("System off.")

            # ---- calibration ----
            if mode == "calibrating":
                guide = collector.feed(ir_val, face["yaw"], face["pitch"], now)
                if collector.is_done():
                    ref = collector.finish()
                    if ref.valid:
                        ref.save()
                        drowsy_det = DrowsinessDetector(ref)
                        inatt_det = InattentionDetector()
                        # The neutral head angle measured during calibration also
                        # becomes the zero point for the forward-attention check.
                        ie.HEAD_YAW_OFFSET_DEG = ref.yaw_offset
                        mode = "running"
                        alert.end_calibration()
                        print(f"Calibration done: open={ref.open_ref:.0f} "
                              f"closed={ref.closed_ref:.0f} "
                              f"(gap {ref.open_ref - ref.closed_ref:.0f}), "
                              f"yaw offset {ref.yaw_offset:+.1f}deg")
                        print("Monitoring started. The first minute tunes the personal threshold.")
                    else:
                        print("Calibration failed (eye open/closed gap too small). Retrying.")
                        collector = CalibrationCollector()

            # ---- running ----
            elif mode == "running":
                # YOLO tracking at its own rate
                if now - last_yolo >= 1.0 / YOLO_HZ:
                    last_yolo = now
                    results = model.track(ext_frame, persist=True, verbose=False)[0]
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

                # forward attention every frame.
                # The engine keeps reporting 1 while an obstacle is still
                # unconfirmed, so the warning clears by itself the moment the
                # driver looks at it or the obstacle leaves the frame.
                inatt_state, iinfo = inatt_det.update(detections, face["yaw"], now=now)
                if inatt_state == 1 and not inatt_warned:
                    for o in iinfo["offenders"]:
                        print(f"[INATTENTION] #{o['id']} {o['class']} "
                              f"{o['angle']:+.0f}deg (level {o['level']})")
                inatt_warned = (inatt_state == 1)

                # drowsiness at the training data rate
                if now - last_infer >= INFER_INTERVAL:
                    last_infer = now
                    drowsy_state, ir_reliable, dinfo = drowsy_det.update(
                        ir=ir_val,
                        yaw=face["yaw"], pitch=face["pitch"],
                        yaw_vel=face["yaw_vel"], pitch_vel=face["pitch_vel"],
                        is_yawning=face["is_yawning"],
                        mouth_open_duration=face["mouth_open_duration"],
                        now=now,
                    )

                # 3 asleep > 2 drowsy > 1 inattention > 0 normal
                final_state = max(drowsy_state, inatt_state)
                alert.update_output(state=final_state, ir_reliable=ir_reliable)

            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            if SHOW_PREVIEW:
                disp = cv2.cvtColor(ext_frame, cv2.COLOR_RGB2BGR)
                draw_overlay(disp, mode, guide, face, ir_val,
                             drowsy_state, inatt_state, final_state,
                             ir_reliable, dinfo, iinfo, detections, fps)
                cv2.imshow("EZdrive", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        picam2.stop()
        face_tracker.close()
        ble_link.stop()
        alert.cleanup()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
