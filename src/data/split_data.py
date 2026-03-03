import os
import shutil
import argparse
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_splits(source_folder, dest_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    classes = [d for d in os.listdir(source_folder) 
               if os.path.isdir(os.path.join(source_folder, d))]
    
    logger.info(f"Found classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(source_folder, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"\nProcessing {class_name}: {len(images)} images")
        
        # First split: train and temp (val + test)
        train, temp = train_test_split(
            images, 
            test_size=(val_ratio + test_ratio),
            random_state=42,
            shuffle=True
        )
        
        # Second split: val and test from temp
        val, test = train_test_split(
            temp,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42,
            shuffle=True
        )
        
        logger.info(f"  Train: {len(train)} images")
        logger.info(f"  Val: {len(val)} images")
        logger.info(f"  Test: {len(test)} images")
        
        # Copy images to respective folders
        for split_name, split_images in [('train', train), ('val', val), ('test', test)]:
            split_class_path = os.path.join(dest_folder, split_name, class_name)
            os.makedirs(split_class_path, exist_ok=True)
            
            for img in tqdm(split_images, desc=f"  Copying to {split_name}"):
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(split_class_path, img)
                )
    
    # Save split information
    split_info = {
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'classes': classes
    }
    
    with open(os.path.join(dest_folder, 'split_info.yaml'), 'w') as f:
        yaml.dump(split_info, f)
    
    logger.info(f"\nDataset splits created successfully in {dest_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/balanced/')
    parser.add_argument('--dest', type=str, default='data/splits/')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()
    
    create_splits(args.source, args.dest, args.train_ratio, args.val_ratio, args.test_ratio)