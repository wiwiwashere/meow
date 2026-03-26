#!/usr/bin/env python
"""
Train binary cat classifier (cat vs not_cat).

Two-phase transfer learning strategy:
  Phase 1 — Frozen backbone, train the head only (fast, stable)
  Phase 2 — Unfreeze top layers, fine-tune end-to-end (lower LR)

Expected data structure:
  data/splits/
      train/
          cat/
          not_cat/
      val/
          cat/
          not_cat/
      test/
          cat/
          not_cat/

Run:
  python scripts/train_bi.py
"""
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.transfer import create_transfer_model, unfreeze_top_layers


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH    = 'data/splits/'
MODEL_DIR    = 'models/'
REPORTS_DIR  = 'reports/'
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_P1    = 10   # Phase 1: head only
EPOCHS_P2    = 20   # Phase 2: fine-tuning (early stopping will cut this short)


# ── Data ──────────────────────────────────────────────────────────────────────

def create_data_generators(data_path=DATA_PATH):
    """
    Create train / val / test generators.
    - Train:    augmentation + rescale
    - Val/Test: rescale only
    """
    train_datagen = ImageDataGenerator(
        # rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    common = dict(
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['cat', 'not_cat'],  # cat=0, not_cat=1
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'), shuffle=True, **common
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(data_path, 'val'), shuffle=False, **common
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(data_path, 'test'), shuffle=False, **common
    )

    print(f"Class mapping : {train_gen.class_indices}")  # {'cat': 0, 'not_cat': 1}
    print(f"Train samples : {train_gen.samples}")
    print(f"Val samples   : {val_gen.samples}")
    print(f"Test samples  : {test_gen.samples}")

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    """Compute class weights to handle mild class imbalance."""
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(weights))
    print(f"Class weights : {class_weights}")
    return class_weights


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(checkpoint_path, patience=5):
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, tag=''):
    """Save accuracy and loss curves to reports/figures/."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title(f'Accuracy {tag}')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'Loss {tag}')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(f'{REPORTS_DIR}figures', exist_ok=True)
    path = f'{REPORTS_DIR}figures/training_history_{tag}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot → {path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_gen):
    """Print metrics and save confusion matrix."""
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )

    test_gen.reset()
    probs = model.predict(test_gen, verbose=1).flatten()
    preds = (probs > 0.5).astype(int)
    true  = test_gen.labels

    metrics = {
        'accuracy' : accuracy_score(true, preds),
        'precision': precision_score(true, preds),
        'recall'   : recall_score(true, preds),
        'f1'       : f1_score(true, preds),
        'roc_auc'  : roc_auc_score(true, probs),
    }

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Not Cat'],
                yticklabels=['Cat', 'Not Cat'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = f'{REPORTS_DIR}figures/confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {cm_path}")

    return metrics


def save_report(metrics):
    os.makedirs(f'{REPORTS_DIR}metrics', exist_ok=True)
    path = f'{REPORTS_DIR}metrics/binary_model_report.txt'
    with open(path, 'w') as f:
        f.write("CAT DETECTOR — BINARY CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<12} {v:.4f}\n")
    print(f"Report saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAT DETECTOR — BINARY CLASSIFICATION TRAINING")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Data
    print("\n[1/6] Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    class_weights = get_class_weights(train_gen)

    # ── 2. Phase 1: Train head only (frozen backbone)
    print("\n[2/6] Phase 1 — training head with frozen backbone...")
    model = create_transfer_model(trainable_base=False)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_P1,
        callbacks=make_callbacks(f'{MODEL_DIR}best_phase1.keras', patience=5),
        class_weight=class_weights,
        verbose=1
    )
    plot_history(history_p1, tag='phase1')

    # ── 3. Phase 2: Unfreeze top layers and fine-tune
    print("\n[3/6] Phase 2 — fine-tuning top backbone layers...")
    model = unfreeze_top_layers(model, num_layers=20)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # Much lower LR
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_P2,
        callbacks=make_callbacks(f'{MODEL_DIR}best_binary_model.keras', patience=5),
        class_weight=class_weights,
        verbose=1
    )
    plot_history(history_p2, tag='phase2')

    # ── 4. Evaluate
    print("\n[4/6] Evaluating on test set...")
    metrics = evaluate_model(model, test_gen)

    # ── 5. Save
    print("\n[5/6] Saving model...")
    model.save(f'{MODEL_DIR}final_binary_model.keras')
    print(f"Model saved → {MODEL_DIR}final_binary_model.keras")

    # ── 6. Report
    print("\n[6/6] Saving report...")
    save_report(metrics)

    print("\nDone! Try: python scripts/predict_single.py path/to/image.jpg")


if __name__ == "__main__":
    main()