import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time

# ===== Settings =====
MODEL_NAME = "yolov8s.pt"  # small model: better accuracy than nano, still runs on Pi
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Classes of interest (COCO dataset class names) for pedestrian/vehicle obstacle detection
TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def main():
    print("Loading YOLO model (first run will download weights)...")
    model = YOLO(MODEL_NAME)

    print("Starting Camera Module 3 (CSI)...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let the camera warm up / auto-exposure settle

    print("Starting detection. Press 'q' to quit.")

    prev_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()  # RGB888 frame as numpy array

            results = model(frame, verbose=False)[0]

            detections = []  # collect (class_name, confidence, box) for this frame

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name not in TARGET_CLASSES:
                    continue
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((class_name, confidence, (x1, y1, x2, y2)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if detections:
                summary = ", ".join(f"{name}({conf:.2f})" for name, conf, _ in detections)
                print(f"Detected: {summary}")

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # picamera2 gives RGB, opencv imshow expects BGR
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("External Camera (CSI) - Obstacle Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()