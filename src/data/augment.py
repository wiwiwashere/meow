import os
import shutil
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from tqdm import tqdm
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def augment_class(input_folder, output_folder, target_count=4000):
    """
    Augment images in a class to reach target count
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images
    images = [f for f in os.listdir(input_folder) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)
    
    logger.info(f"Class {os.path.basename(input_folder)}:")
    logger.info(f"  Current count: {current_count}")
    logger.info(f"  Target count: {target_count}")
    
    # Copy original images
    logger.info("  Copying original images...")
    for img in tqdm(images, desc="  Copying"):
        shutil.copy2(
            os.path.join(input_folder, img),
            os.path.join(output_folder, f"orig_{img}")
        )
    
    # Check if augmentation needed
    aug_needed = target_count - current_count
    if aug_needed <= 0:
        logger.info(f"  No augmentation needed (already have {current_count} images)")
        return
    
    # Setup augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Generate augmented images
    logger.info(f"  Generating {aug_needed} augmented images...")
    aug_count = 0
    
    while aug_count < aug_needed:
        # Randomly select an image to augment
        img_name = random.choice(images)
        img_path = os.path.join(input_folder, img_name)
        
        try:
            # Load and augment image
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            # Generate augmented version
            for batch in datagen.flow(x, batch_size=1):
                aug_img = batch[0].astype('uint8')
                aug_img_name = f"aug_{aug_count:04d}_{img_name}"
                aug_img_path = os.path.join(output_folder, aug_img_name)
                
                Image.fromarray(aug_img).save(aug_img_path)
                
                aug_count += 1
                if aug_count >= aug_needed:
                    break
                if aug_count % 100 == 0:
                    logger.info(f"    Generated {aug_count}/{aug_needed}")
                    
        except Exception as e:
            logger.warning(f"    Error augmenting {img_name}: {e}")
            continue
    
    final_count = len(os.listdir(output_folder))
    logger.info(f"  Final count: {final_count} images")

def balance_dataset(processed_path, balanced_path, target_count=4000):
    """
    Balance all classes to target_count
    """
    classes = ['cat', 'dog', 'felidae']
    
    for class_name in classes:
        input_folder = os.path.join(processed_path, class_name)
        output_folder = os.path.join(balanced_path, class_name)
        
        if os.path.exists(input_folder):
            augment_class(input_folder, output_folder, target_count)
        else:
            logger.warning(f"Class folder {input_folder} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, default='data/processed/')
    parser.add_argument('--balanced_path', type=str, default='data/balanced/')
    parser.add_argument('--target_count', type=int, default=4000)
    args = parser.parse_args()
    
    balance_dataset(args.processed_path, args.balanced_path, args.target_count)