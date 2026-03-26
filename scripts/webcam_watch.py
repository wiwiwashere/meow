#!/usr/bin/env python
"""
Live webcam cat detector with SMS alerts.

Captures frames from your webcam, runs the classifier on each,
and sends an SMS when a cat is detected. Includes a cooldown to
avoid spamming alerts for the same cat.

Usage:
  python scripts/webcam_watch.py
  python scripts/webcam_watch.py --cooldown 60   # seconds between alerts
  python scripts/webcam_watch.py --threshold 0.7 # higher = more certain before alerting
  python scripts/webcam_watch.py --no-alert       # run detection without SMS

Press Q to quit.

Install:
  pip install opencv-python
"""
import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.notifications.twilio_alert import send_cat_alert

IMG_SIZE      = (224, 224)
DEFAULT_MODEL = 'models/best_binary_model.keras'


def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run: python scripts/train_bi.py")
        sys.exit(1)
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)


def preprocess_frame(frame):
    """Convert an OpenCV BGR frame to model input."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    arr   = resized.astype(np.float32)
    arr   = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def draw_overlay(frame, label, conf, is_cat):
    """Draw prediction overlay on the frame."""
    color = (0, 200, 100) if is_cat else (100, 100, 200)
    text  = f"{label}  {conf:.1%}"
    cv2.rectangle(frame, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Live webcam cat detector')
    parser.add_argument('--model',     type=str,   default=DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for cat detection (default: 0.5)')
    parser.add_argument('--cooldown',  type=int,   default=30,
                        help='Seconds to wait between SMS alerts (default: 30)')
    parser.add_argument('--no-alert',  action='store_true',
                        help='Disable SMS alerts')
    parser.add_argument('--camera',    type=int,   default=0,
                        help='Camera index (default: 0)')
    args = parser.parse_args()

    model = load_model(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    print(f"\nWebcam cat detector running")
    print(f"Threshold : {args.threshold}")
    print(f"Cooldown  : {args.cooldown}s between alerts")
    print(f"Alerts    : {'disabled' if args.no_alert else 'enabled (Twilio SMS)'}")
    print("Press Q to quit\n")

    last_alert_time = 0
    frame_count     = 0
    PREDICT_EVERY   = 10  # run model every N frames to keep it responsive

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        frame_count += 1

        # Run prediction every N frames
        if frame_count % PREDICT_EVERY == 0:
            image_array      = preprocess_frame(frame)
            prob             = float(model.predict(image_array, verbose=0)[0][0])
            is_cat           = prob < args.threshold
            label            = "Cat" if is_cat else "Not a cat"
            conf             = (1 - prob) if is_cat else prob

            # Print to terminal
            status = f"\r{label}  ({conf:.1%})" + (" " * 20)
            print(status, end="", flush=True)

            # Send alert if cat detected and cooldown has passed
            now = time.time()
            if is_cat and not args.no_alert and (now - last_alert_time) > args.cooldown:
                print(f"\nCat detected at {conf:.1%} confidence — sending SMS...")
                success = send_cat_alert(
                    confidence=conf,
                    source="webcam",
                )
                if success:
                    print("SMS sent!")
                    last_alert_time = now
                else:
                    print("SMS failed — check your .env file")

            # Store for overlay
            last_label = label
            last_conf  = conf
            last_is_cat = is_cat
        else:
            # Use last prediction for overlay between frames
            try:
                label, conf, is_cat = last_label, last_conf, last_is_cat
            except NameError:
                label, conf, is_cat = "Starting...", 0.0, False

        # Draw overlay and show frame
        frame = draw_overlay(frame, label, conf, is_cat)
        cv2.imshow("Cat Detector — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam stopped.")


if __name__ == "__main__":
    main()