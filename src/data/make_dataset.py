import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_raw_data(raw_path: str, processed_path: str):
    """
    Copy raw data to processed folder (first step before augmentation)
    """
    classes = ['cat', 'dog', 'felidae']
    
    for class_name in classes:
        source_dir = os.path.join(raw_path, class_name)
        dest_dir = os.path.join(processed_path, class_name)
        
        # Create destination directory
        os.makedirs(dest_dir, exist_ok=True)
        
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory {source_dir} doesn't exist")
            continue
        
        # Copy images
        images = os.listdir(source_dir)
        logger.info(f"Copying {len(images)} {class_name} images")
        
        for img in tqdm(images, desc=f"Copying {class_name}"):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(
                    os.path.join(source_dir, img),
                    os.path.join(dest_dir, img)
                )
    
    logger.info("Raw data organized successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='data/raw/')
    parser.add_argument('--processed_path', type=str, default='data/processed/')
    args = parser.parse_args()
    
    organize_raw_data(args.raw_path, args.processed_path)