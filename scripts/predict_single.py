#!/usr/bin/env python
"""
Simple script to test if an image contains a cat
Usage: python scripts/predict_binary.py path/to/image.jpg
"""
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array


def display_image(img_path, title):
    """Display the image being tested"""
    img = mpimg.imread(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def is_cat(img_path, model_path='models/final_binary_model.h5'):
    """
    Check if image contains a cat
    Returns: dict with 'is_cat' (bool) and 'confidence' (float)
    """
    
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(img_path)
    
    # Predict
    probability = model.predict(img_array, verbose=0)[0][0]
    
    # Binary decision (cat = probability > 0.5)
    is_cat = probability < 0.5  # Because cat is class 0, not_cat is class 1
    
    result = {
        'is_cat': bool(is_cat),
        'probability': float(probability),
        'confidence': float(1 - probability) if is_cat else float(probability)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Detect if an image contains a cat')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--model', default='models/final_binary_model.h5',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    try:
        result = is_cat(args.image_path, args.model)
        
        print("\n")
        print("="*50)
        print("CAT DETECTOR RESULT")
        print("="*50)
        print(f"Image: {args.image_path}")
        print("-"*50)
        
        if result['is_cat']:
            display_image(args.image_path, title="This is a CAT!")
            print(f"Confidence: {result['confidence']:.2%}")          
        else:
            display_image(args.image_path, title="This is NOT a CAT!")
            print(f"Confidence: {result['confidence']:.2%}")

        print(f"Raw probability: {result['probability']:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()