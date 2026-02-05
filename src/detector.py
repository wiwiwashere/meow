'''
This module handles object detection functionalities for the application.
true if cat
false otherwise
'''

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional

class CatDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the cat detector.
        """
        # Start with simple Haar Cascade for initial testing
        model_path = None
        self.cat_model = None
    
    def detect_cat(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[bool, float, List]:
        """
        Detect if there's a cat in the image.
        Returns: (detected, confidence, bounding_boxes)

        bounding boxes: Bounding boxes are rectangular, defined by (𝑥,𝑦) coordinates, that isolate objects in images for detection
        """
        if self.cat_model is None:
            # Placeholder
            return False, 0.0, []
        
        # Detect cats
        cats = None

        # Detection logic
        
        return False, 0.0, []
    
    # visulize results

if __name__ == "__main__":
    # Test the detector on a sample image
    detector = CatDetector()
    
    test_image = cv2.imread("../data/test_images/c1.jpeg")
    detected, confidence, boxes = detector.detect_cat(test_image)
    
    print(f"Detected: {detected}")
    print(f"Confidence: {confidence:.2%}")