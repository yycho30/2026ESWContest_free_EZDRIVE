import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
import csv
import serial
import datetime

# ===== Settings =====
MODEL_PATH = "face_landmarker.task"   # path to the downloaded .task model file
MAR_THRESHOLD = 0.6                   # threshold for "mouth open"
YAWN_HOLD_TIME = 1.5                  # seconds mouth must stay open to count as a yawn

SERIAL_PORT = "/dev/ttyACM0"          # Arduino Uno USB serial port
BAUD_RATE = 9600

LOG_HZ = 10.0                         # CSV logging rate (rows per second)
LOG_INTERVAL = 1.0 / LOG_HZ
CSV_DIR = "."                         # folder where session CSV files are saved

CSV_HEADER = [
    "timestamp", "yaw_deg", "pitch_deg",
    "yaw_angular_velocity_deg_s", "pitch_angular_velocity_deg_s",
    "is_yawning", "yawn_count", "mouth_open_duration_s", "ir_value"
]

# Generic 3D face model points (mm): nose tip, chin, eye outer corners, mouth corners
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


# ===== Shared state for the serial reader thread =====
class IRReader:
    """Reads IR values from the Arduino in the background and always keeps
    the most recent value available, without blocking the camera loop."""

    def __init__(self, port, baud):
        self.latest_value = None
        self._lock = threading.Lock()
        self._stop_flag = False
        self._ser = None

        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            print(f"IR serial connected on {port}")
        except serial.SerialException as e:
            print(f"Could not open IR serial port: {e}")
            print("Continuing without IR data (IR column will stay empty).")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        if self._ser is None:
            return
        while not self._stop_flag:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    with self._lock:
                        self.latest_value = line
            except serial.SerialException:
                break

    def get_latest(self):
        with self._lock:
            return self.latest_value

    def stop(self):
        self._stop_flag = True
        if self._ser is not None:
            self._ser.close()


def calc_mar(landmarks, img_w, img_h):
    top = np.array([landmarks[MOUTH_TOP].x * img_w, landmarks[MOUTH_TOP].y * img_h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * img_w, landmarks[MOUTH_BOTTOM].y * img_h])
    left = np.array([landmarks[MOUTH_LEFT].x * img_w, landmarks[MOUTH_LEFT].y * img_h])
    right = np.array([landmarks[MOUTH_RIGHT].x * img_w, landmarks[MOUTH_RIGHT].y * img_h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[LANDMARK_IDS["nose_tip"]].x * img_w, landmarks[LANDMARK_IDS["nose_tip"]].y * img_h),
        (landmarks[LANDMARK_IDS["chin"]].x * img_w, landmarks[LANDMARK_IDS["chin"]].y * img_h),
        (landmarks[LANDMARK_IDS["left_eye_outer"]].x * img_w, landmarks[LANDMARK_IDS["left_eye_outer"]].y * img_h),
        (landmarks[LANDMARK_IDS["right_eye_outer"]].x * img_w, landmarks[LANDMARK_IDS["right_eye_outer"]].y * img_h),
        (landmarks[LANDMARK_IDS["mouth_left"]].x * img_w, landmarks[LANDMARK_IDS["mouth_left"]].y * img_h),
        (landmarks[LANDMARK_IDS["mouth_right"]].x * img_w, landmarks[LANDMARK_IDS["mouth_right"]].y * img_h),
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()

    pitch = pitch if pitch <= 90 else pitch - 180
    if pitch < -90:
        pitch += 180

    return yaw, pitch


def start_new_csv():
    """Create a new timestamped CSV file and return (file_handle, writer, path)."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{CSV_DIR}/session_{ts}.csv"
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    return f, writer, path


def main():
    # ----- Set up MediaPipe Tasks FaceLandmarker -----
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # ----- Set up IR background reader -----
    ir_reader = IRReader(SERIAL_PORT, BAUD_RATE)

    # ----- Set up camera -----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Check device index (0).")
        ir_reader.stop()
        return

    prev_yaw = None
    prev_pitch = None
    prev_time = None

    yawn_count = 0
    is_yawning = False
    mouth_open_start_time = None
    mouth_open_duration = 0.0

    last_log_time = 0.0

    # ----- Recording state (controlled by 's' key) -----
    is_recording = False
    csv_file = None
    csv_writer = None
    csv_path = None

    print("Press 's' to start/stop recording. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame.")
                break

            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_time = time.time()

            yaw = pitch = None
            angular_velocity_yaw = 0.0
            angular_velocity_pitch = 0.0
            mar = 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                yaw, pitch = estimate_head_pose(landmarks, img_w, img_h)

                if yaw is not None and prev_yaw is not None and prev_time is not None:
                    dt = current_time - prev_time
                    if dt > 0:
                        angular_velocity_yaw = (yaw - prev_yaw) / dt
                        angular_velocity_pitch = (pitch - prev_pitch) / dt

                if yaw is not None:
                    prev_yaw = yaw
                    prev_pitch = pitch
                    prev_time = current_time

                mar = calc_mar(landmarks, img_w, img_h)
                mouth_is_open = mar > MAR_THRESHOLD

                if mouth_is_open:
                    if mouth_open_start_time is None:
                        mouth_open_start_time = current_time
                    mouth_open_duration = current_time - mouth_open_start_time

                    if mouth_open_duration >= YAWN_HOLD_TIME and not is_yawning:
                        is_yawning = True
                        yawn_count += 1
                else:
                    mouth_open_start_time = None
                    mouth_open_duration = 0.0
                    is_yawning = False

                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawn: {is_yawning} Count: {yawn_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"MouthOpen: {mouth_open_duration:.2f}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----- IR value overlay (always shown, regardless of recording state) -----
            current_ir_value = ir_reader.get_latest()
            ir_display = current_ir_value if current_ir_value is not None else "N/A"
            cv2.putText(frame, f"IR: {ir_display}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----- Recording status overlay -----
            status_text = "RECORDING" if is_recording else "STOPPED (press 's' to start)"
            status_color = (0, 0, 255) if is_recording else (200, 200, 200)
            cv2.putText(frame, status_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # ----- Log to CSV at fixed rate, only while recording -----
            if is_recording and (current_time - last_log_time >= LOG_INTERVAL):
                last_log_time = current_time
                ir_value = ir_reader.get_latest()

                csv_writer.writerow([
                    f"{current_time:.3f}",
                    f"{yaw:.2f}" if yaw is not None else "",
                    f"{pitch:.2f}" if pitch is not None else "",
                    f"{angular_velocity_yaw:.2f}",
                    f"{angular_velocity_pitch:.2f}",
                    is_yawning,
                    yawn_count,
                    f"{mouth_open_duration:.2f}",
                    ir_value if ir_value is not None else "",
                ])
                csv_file.flush()

            cv2.imshow("Face Pose & Yawn Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if not is_recording:
                    # ----- Start recording: open a new CSV -----
                    csv_file, csv_writer, csv_path = start_new_csv()
                    is_recording = True
                    last_log_time = 0.0  # log immediately on the next loop
                    print(f"Recording started -> {csv_path}")
                else:
                    # ----- Stop recording: close the CSV -----
                    is_recording = False
                    csv_file.close()
                    print(f"Recording stopped. Saved to {csv_path}")
                    csv_file = None
                    csv_writer = None
                    csv_path = None

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        ir_reader.stop()
        if csv_file is not None:
            csv_file.close()
            print(f"Recording stopped (program exit). Saved to {csv_path}")


if __name__ == "__main__":
    main()