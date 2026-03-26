"""
Augment the training set only — run AFTER split_data.py.

Correct order:
  1. split_data.py   →  data/splits/train | val | test
  2. augment.py      →  augments data/splits/train only
  3. train_bi.py     →  trains the model

Usage:
  python scripts/augment.py
  python scripts/augment.py --train_path data/splits/train --target_count 4000
"""
import os
import shutil
import argparse
import random
import logging
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Augmentation pipeline (training only)
DATAGEN = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)


def augment_class(class_folder, target_count=4000):
    """
    Augment a single class folder in-place until it reaches target_count.
    Original images are kept; augmented ones are added alongside them.

    Args:
        class_folder:  Path to e.g. data/splits/train/cat
        target_count:  Total images to reach (originals + augmented)
    """
    images = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    current_count = len(images)
    class_name = os.path.basename(class_folder)

    logger.info(f"\n[{class_name}] {current_count} images → target {target_count}")

    if current_count >= target_count:
        logger.info(f"[{class_name}] Already at target, skipping.")
        return

    aug_needed = target_count - current_count
    logger.info(f"[{class_name}] Generating {aug_needed} augmented images...")

    aug_count = 0
    with tqdm(total=aug_needed, desc=f"  Augmenting {class_name}") as pbar:
        while aug_count < aug_needed:
            img_name = random.choice(images)
            img_path = os.path.join(class_folder, img_name)

            try:
                img = load_img(img_path)
                x = img_to_array(img).reshape((1,) + img_to_array(img).shape)

                for batch in DATAGEN.flow(x, batch_size=1):
                    aug_array = batch[0].astype('uint8')
                    save_name = f"aug_{aug_count:05d}_{img_name}"
                    Image.fromarray(aug_array).save(
                        os.path.join(class_folder, save_name)
                    )
                    aug_count += 1
                    pbar.update(1)
                    break  # one augmentation per source image per loop

                if aug_count >= aug_needed:
                    break

            except Exception as e:
                logger.warning(f"  Skipped {img_name}: {e}")
                continue

    final_count = len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    logger.info(f"[{class_name}] Done — {final_count} total images")


def augment_train_split(train_path, target_count=4000):
    """
    Augment all class folders inside the train split.

    Expected structure:
        train_path/
            cat/
            not_cat/

    Args:
        train_path:    Path to training split (data/splits/train)
        target_count:  Target images per class
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path not found: {train_path}")

    classes = [
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ]

    if not classes:
        raise ValueError(f"No class folders found in {train_path}")

    logger.info(f"Found classes: {classes}")
    logger.info("Augmenting training split only (val/test are untouched)")

    for class_name in classes:
        class_folder = os.path.join(train_path, class_name)
        augment_class(class_folder, target_count)

    logger.info("\nAugmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/splits/train',
                        help='Path to training split folder')
    parser.add_argument('--target_count', type=int, default=4000,
                        help='Target image count per class')
    args = parser.parse_args()

    augment_train_split(args.train_path, args.target_count)