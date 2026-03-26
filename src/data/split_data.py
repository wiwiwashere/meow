"""
Split raw images into train / val / test sets.

Merges dog and felidae into a single 'not_cat' class, producing:

  data/splits/
      train/  cat/  not_cat/
      val/    cat/  not_cat/
      test/   cat/  not_cat/

Usage:
  python scripts/split_data.py
  python scripts/split_data.py --source data/raw --dest data/splits
"""
import os
import shutil
import argparse
import logging
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Map raw folder names -> binary class name
CLASS_MAP = {
    'cat'     : 'cat',
    'dog'     : 'not_cat',
    'felidae' : 'not_cat',
}


def collect_images(source_folder):
    """
    Walk source_folder and return a dict:
        { 'cat': [...paths...], 'not_cat': [...paths...] }
    """
    collected = {'cat': [], 'not_cat': []}

    for raw_class, binary_class in CLASS_MAP.items():
        class_path = os.path.join(source_folder, raw_class)
        if not os.path.exists(class_path):
            logger.warning(f"Folder not found, skipping: {class_path}")
            continue

        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        collected[binary_class].extend(images)
        logger.info(f"  {raw_class:10s} ({binary_class}) → {len(images)} images")

    return collected


def split_and_copy(image_paths, dest_folder, class_name,
                   train_ratio, val_ratio, test_ratio):
    """Split a list of image paths and copy into dest/split/class_name/."""

    train, temp = train_test_split(
        image_paths,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        shuffle=True
    )
    val, test = train_test_split(
        temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42,
        shuffle=True
    )

    logger.info(f"  [{class_name}] train={len(train)}  val={len(val)}  test={len(test)}")

    for split_name, split_images in [('train', train), ('val', val), ('test', test)]:
        out_dir = os.path.join(dest_folder, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for src_path in tqdm(split_images, desc=f"    → {split_name}/{class_name}", leave=False):
            filename = os.path.basename(src_path)
            shutil.copy2(src_path, os.path.join(out_dir, filename))

    return {'train': len(train), 'val': len(val), 'test': len(test)}


def create_splits(source_folder, dest_folder,
                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1.0"

    logger.info(f"\nSource : {source_folder}")
    logger.info(f"Dest   : {dest_folder}")
    logger.info(f"Split  : {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test\n")

    # Collect and merge into binary classes
    logger.info("Collecting images...")
    collected = collect_images(source_folder)

    for cls, paths in collected.items():
        logger.info(f"  {cls}: {len(paths)} total images")

    # Split each binary class
    logger.info("\nSplitting and copying...")
    stats = {}
    for class_name, image_paths in collected.items():
        if not image_paths:
            logger.warning(f"No images found for class '{class_name}', skipping.")
            continue
        stats[class_name] = split_and_copy(
            image_paths, dest_folder, class_name,
            train_ratio, val_ratio, test_ratio
        )

    # Save a summary yaml for reference
    summary = {
        'split_ratios': {
            'train': train_ratio,
            'val'  : val_ratio,
            'test' : test_ratio,
        },
        'classes'      : list(stats.keys()),
        'class_map'    : CLASS_MAP,
        'counts'       : stats,
    }
    summary_path = os.path.join(dest_folder, 'split_info.yaml')
    os.makedirs(dest_folder, exist_ok=True)
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    logger.info(f"\nDone. Split info saved to {summary_path}")
    logger.info("Next step: python scripts/augment.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/raw',
                        help='Folder with cat/, dog/, felidae/ subfolders')
    parser.add_argument('--dest', type=str, default='data/splits',
                        help='Output folder for train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--test_ratio',  type=float, default=0.1)
    args = parser.parse_args()

    create_splits(
        args.source, args.dest,
        args.train_ratio, args.val_ratio, args.test_ratio
    )