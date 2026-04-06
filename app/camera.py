"""
Webcam stream with live cat detection.
Runs the model on every Nth frame and overlays the result.
"""
import cv2
import time
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE     = (224, 224)
PREDICT_EVERY = 10   # run model every N frames


class CameraStream:
    def __init__(self, model_path: str, camera_index: int = 0, threshold: float = 0.5):
        self.threshold = threshold
        self._lock     = threading.Lock()
        self._state    = {
            "label"     : "Starting...",
            "confidence": 0.0,
            "is_cat"    : False,
            "timestamp" : time.time(),
        }

        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded.")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def _predict(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
        arr     = preprocess_input(resized)
        prob    = float(self.model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
        is_cat  = prob < self.threshold
        conf    = (1 - prob) if is_cat else prob
        label   = "Cat" if is_cat else "Not a cat"
        return label, conf, is_cat

    def _draw_overlay(self, frame, label, conf, is_cat):
        color  = (80, 200, 120) if is_cat else (100, 100, 220)
        text   = f"{label}  {conf*100:.1f}%"
        h, w   = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        return frame

    def generate_frames(self):
        """Generator — yields MJPEG frames for StreamingResponse."""
        frame_count   = 0
        last_label    = "Starting..."
        last_conf     = 0.0
        last_is_cat   = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % PREDICT_EVERY == 0:
                label, conf, is_cat = self._predict(frame)
                last_label, last_conf, last_is_cat = label, conf, is_cat

                with self._lock:
                    self._state = {
                        "label"     : label,
                        "confidence": conf,
                        "is_cat"    : is_cat,
                        "timestamp" : time.time(),
                    }

            frame = self._draw_overlay(frame, last_label, last_conf, last_is_cat)

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )