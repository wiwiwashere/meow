import cv2
import os
import glob
import numpy as np
from detector import CatDetector, visualize_detection

def test_on_all_images():
    """Test the detector on all images in the test_images folder."""
    detector = CatDetector()
    
    # Get all image files
    image_paths = glob.glob("../data/test_images/*.jpeg") + \
                  glob.glob("../data/test_images/*.jpg")
    
    results = []
    
    for img_path in image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Read image
        image = cv2.imread(img_path)
        
        # Detect cat
        detected, confidence, boxes = detector.detect_cat(image)
        
        # Save visualization
        output_dir = "../data/test_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}")
        
        result_img = visualize_detection(image, detected, confidence, boxes, output_path)
        print(f"Saved result to: {output_path}")
        
        results.append({
            'filename': os.path.basename(img_path),
            'detected': detected,
            'confidence': confidence,
            'boxes': boxes
        })
    
    return results

if __name__ == "__main__":
    test_on_all_images()