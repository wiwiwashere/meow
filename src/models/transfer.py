import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def create_transfer_model(input_shape=(224, 224, 3), trainable_base=False):
    """
    Binary cat classifier using pretrained EfficientNetB0.

    Args:
        input_shape:     Image dimensions, default 224x224 RGB
        trainable_base:  False = frozen backbone (Phase 1)
                         True  = full fine-tuning (Phase 2)
    """
    # Load pretrained backbone (no top classifier)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary output

    model = models.Model(inputs, outputs, name='cat_classifier_transfer')
    return model


def unfreeze_top_layers(model, num_layers=20):
    """
    Unfreeze the top N layers of the backbone for fine-tuning (Phase 2).
    Call this after Phase 1 training converges.

    Args:
        model:      The transfer model returned by create_transfer_model()
        num_layers: How many layers from the end of the backbone to unfreeze
    """
    base_model = model.layers[1]  # EfficientNetB0 is the second layer
    base_model.trainable = True

    # Freeze everything except the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze top {trainable} layers of backbone for fine-tuning")
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = create_transfer_model()
    model.summary()