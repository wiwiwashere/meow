# #!/usr/bin/env python
# """
# Run the trained cat classifier on a single image.

# Usage:
#   python scripts/predict_single.py path/to/image.jpg
#   python scripts/predict_single.py path/to/image.jpg --model models/best_binary_model.keras
#   python scripts/predict_single.py path/to/image.jpg --threshold 0.6
# """
# import sys
# import os
# import argparse
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# IMG_SIZE     = (224, 224)
# DEFAULT_MODEL = 'models/best_binary_model.keras'


# def load_model(model_path):
#     if not os.path.exists(model_path):
#         # Also try .h5 fallback
#         h5_path = model_path.replace('.keras', '.h5')
#         if os.path.exists(h5_path):
#             model_path = h5_path
#         else:
#             print(f"Error: Model not found at {model_path}")
#             print("Have you trained the model yet? Run: python scripts/train_bi.py")
#             sys.exit(1)

#     print(f"Loading model from {model_path}...")
#     return tf.keras.models.load_model(model_path)


# def preprocess_image(image_path):
#     if not os.path.exists(image_path):
#         print(f"Error: Image not found at {image_path}")
#         sys.exit(1)

#     img = load_img(image_path, target_size=IMG_SIZE)
#     arr = img_to_array(img)
#     arr = tf.keras.applications.efficientnet.preprocess_input(arr)
#     return np.expand_dims(arr, axis=0)


# def predict(model, image_array, threshold=0.5):
#     prob = float(model.predict(image_array, verbose=0)[0][0])
#     # prob close to 0 → cat, close to 1 → not_cat  (class_indices: cat=0, not_cat=1)
#     is_cat = prob < threshold
#     label = "Cat" if is_cat else "Not a cat"
#     conf = (1 - prob) if is_cat else prob
#     return label, conf, prob


# def main():
#     parser = argparse.ArgumentParser(description='Cat vs Not-Cat classifier')
#     parser.add_argument('image_path', type=str,
#                         help='Path to the image to classify')
#     parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
#                         help=f'Path to trained model (default: {DEFAULT_MODEL})')
#     parser.add_argument('--threshold', type=float, default=0.5,
#                         help='Decision threshold (default: 0.5)')
#     args = parser.parse_args()

#     model = load_model(args.model)
#     image_array = preprocess_image(args.image_path)
#     label, conf, raw_prob = predict(model, image_array, args.threshold)

#     print(f"\nImage    : {args.image_path}")
#     print(f"Result   : {label}")
#     print(f"Confidence: {conf:.1%}")
#     print(f"Raw score : {raw_prob:.4f}  (0=cat, 1=not_cat, threshold={args.threshold})")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
"""
Run the trained cat classifier on a single image.
Sends an SMS via Twilio if a cat is detected.

Usage:
  python scripts/predict_single.py path/to/image.jpg
  python scripts/predict_single.py path/to/image.jpg --no-alert
  python scripts/predict_single.py path/to/image.jpg --threshold 0.6
"""
import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.notifications.twilio_alert import send_cat_alert

IMG_SIZE      = (224, 224)
DEFAULT_MODEL = 'models/best_binary_model.keras'


def load_model(model_path):
    if not os.path.exists(model_path):
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            print(f"Error: Model not found at {model_path}")
            print("Have you trained the model yet? Run: python scripts/train_bi.py")
            sys.exit(1)
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, image_array, threshold=0.5):
    prob   = float(model.predict(image_array, verbose=0)[0][0])
    is_cat = prob < threshold
    label  = "Cat" if is_cat else "Not a cat"
    conf   = (1 - prob) if is_cat else prob
    return label, conf, prob, is_cat


def main():
    parser = argparse.ArgumentParser(description='Cat vs Not-Cat classifier')
    parser.add_argument('image_path',  type=str, help='Path to image')
    parser.add_argument('--model',     type=str, default=DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-alert',  action='store_true',
                        help='Skip SMS alert even if cat is detected')
    args = parser.parse_args()

    model                         = load_model(args.model)
    image_array                   = preprocess_image(args.image_path)
    label, conf, raw_prob, is_cat = predict(model, image_array, args.threshold)

    print(f"\nImage     : {args.image_path}")
    print(f"Result    : {label}")
    print(f"Confidence: {conf:.1%}")
    print(f"Raw score : {raw_prob:.4f}  (0=cat, 1=not_cat, threshold={args.threshold})")

    if is_cat and not args.no_alert:
        print("\nCat detected — sending SMS alert...")
        success = send_cat_alert(
            confidence=conf,
            source="predict_single",
            image_path=args.image_path
        )
        print("SMS sent!" if success else "SMS failed — check your .env file")
    elif not is_cat:
        print("\nNo cat detected — no alert sent")
    else:
        print("\n(alerts disabled via --no-alert)")


if __name__ == "__main__":
    main()