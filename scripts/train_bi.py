#!/usr/bin/env python
"""
Train a binary classifier (cat vs not-cat)

This will:
- Load data
- Create a CNN model
- Train for 20 epochs
- Show accuracy and loss plots
- Save the model

Run: python scripts/train_bi.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
from src.models.cnn_model import create_cnn_model

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_binary_cnn(input_shape=(224, 224, 3)):
    """Create a CNN for binary classification"""
    
    model = models.Sequential([
        # First block
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        # Binary output (sigmoid for probability)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def create_data_generators(data_path='data/splits/'):
    """Create data generators for binary classification"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',  # Binary labels
        classes=['cat', 'not_cat'],  # Explicitly set order: cat=0, not_cat=1
        shuffle=True
    )
    
    val_gen = valid_datagen.flow_from_directory(
        os.path.join(data_path, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['cat', 'not_cat'],
        shuffle=False
    )
    
    test_gen = valid_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['cat', 'not_cat'],
        shuffle=False
    )
    
    print("\n")
    print(f"Class mapping: {train_gen.class_indices}")  # Should be {'cat': 0, 'not_cat': 1}
    
    return train_gen, val_gen, test_gen


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/binary_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_gen):
    """Detailed model evaluation"""
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen)
    predictions = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    true_labels = test_gen.labels
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import seaborn as sns
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n")
    print("="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}") # When it says cat, how often is it right?
    print(f"Recall:    {recall:.3f}") # How many actual cats did it find?
    print(f"F1-Score:  {f1:.3f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cat', 'Not Cat'],
                yticklabels=['Cat', 'Not Cat'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    print("="*60)
    print("BEGIN CAT DETECTOR TRAINING (Binary Classification)")
    print("="*60)
    
    # 1. Load and prepare data
    print("\n")
    print("Step 1: Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # 2. Create model
    print("\n")
    print("Step 2: Creating binary CNN model...")
    model = create_cnn_model()
    
    # Compile with binary metrics
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 'Precision', 
                 'Recall']
    )
    
    model.summary()
    
    # 3. Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_binary_model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 4. Train model
    print("\n")
    print("Step 3: Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10, # Start with 10 epochs for binary classification
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Plot training history
    print("\n")
    print("Step 4: Plotting training history...")
    plot_training_history(history)
    
    # 6. Evaluate on test set
    print("\n")
    print("Step 5: Evaluating on test set...")
    metrics = evaluate_model(model, test_gen)
    
    # 7. Save final model
    model.save('models/final_binary_model.h5')
    print("\n")
    print("Model saved to models/final_binary_model.h5")
    
    # 8. Save metrics to file
    with open('reports/metrics/binary_model_report.txt', 'w') as f:
        f.write("CAT DETECTOR - BINARY CLASSIFICATION REPORT\n")
        f.write("="*50)
        print("\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall:    {metrics['recall']:.3f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.3f}\n")
    
    print("\n")
    print("Training complete! Your cat detector is ready!")
    print("Try it out: python scripts/predict_single.py path/to/your/image.jpg")

if __name__ == "__main__":
    main()