import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path='configs/data_config.yaml'):
        # Convert to absolute path
        if not os.path.isabs(config_path):
            # Try multiple possible locations
            possible_paths = [
                config_path,
                os.path.join(os.getcwd(), config_path),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        logger.info(f"Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
            logger.info(f"Config keys: {list(self.config.keys())}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self.get_default_config()
            logger.info("Using default config")
    
    def get_default_config(self):
        """Return default configuration if file not found"""
        return {
            'image_params': {
                'target_size': [224, 224],
                'batch_size': 32
            },
            'augmentation': {
                'rotation_range': 30,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'shear_range': 0.2,
                'zoom_range': 0.2,
                'horizontal_flip': True,
                'brightness_range': [0.8, 1.2],
                'fill_mode': 'nearest'
            }
        }
    
    def create_data_generators(self, data_path='data/splits/'):
        """
        Create train, validation, and test data generators
        """
        # Check if data path exists
        if not os.path.exists(data_path):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            alt_path = os.path.join(project_root, data_path)
            if os.path.exists(alt_path):
                data_path = alt_path
            else:
                raise FileNotFoundError(f"Data path not found: {data_path}")
        
        logger.info(f"Using data path: {data_path}")
        
        # Get config values with defaults
        target_size = tuple(self.config.get('image_params', {}).get('target_size', [224, 224]))
        batch_size = self.config.get('image_params', {}).get('batch_size', 32)
        aug_config = self.config.get('augmentation', {})
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=aug_config.get('rotation_range', 30),
            width_shift_range=aug_config.get('width_shift_range', 0.2),
            height_shift_range=aug_config.get('height_shift_range', 0.2),
            shear_range=aug_config.get('shear_range', 0.2),
            zoom_range=aug_config.get('zoom_range', 0.2),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            brightness_range=aug_config.get('brightness_range', [0.8, 1.2]),
            fill_mode=aug_config.get('fill_mode', 'nearest')
        )
        
        # Only rescaling for validation and test
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # Check if train directory exists
        train_path = os.path.join(data_path, 'train')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train directory not found at: {train_path}")
        
        # Create generators
        logger.info(f"Creating train generator from: {train_path}")
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_path = os.path.join(data_path, 'val')
        logger.info(f"Creating validation generator from: {val_path}")
        validation_generator = valid_datagen.flow_from_directory(
            val_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_path = os.path.join(data_path, 'test')
        logger.info(f"Creating test generator from: {test_path}")
        test_generator = valid_datagen.flow_from_directory(
            test_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Log class indices
        logger.info(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, validation_generator, test_generator