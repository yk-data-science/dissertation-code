from keras import layers, models
import tensorflow as tf

def create_simple_cnn(input_shape, num_classes, augmentation):
    """
    Build a simple CNN model with data augmentation.
    """
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs) # Apply data augmentation
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

def create_cnn_for_oct_biomarker(input_shape, num_classes, augmentation):
    """
    Build a CNN model tailored for OCT biomarker detection with data augmentation.
    This model is deeper and more complex than the simple CNN.
    """
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs) # Apply data augmentation
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x) # Added BatchNormalization layer
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x) # Increased dropout rate for regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

_cached_base_model = None
def get_cached_base_model(input_shape):
    """
    Cache and return a pre-trained MobileNetV2 model.
    """
    global _cached_base_model # Use global variable to store cached model
    if _cached_base_model is None:
        base_model = tf.keras.applications.MobileNetV2( # Load MobileNetV2 only once
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        _cached_base_model = base_model
    return _cached_base_model

def create_transfer_model(input_shape, num_classes, augmentation, unfreeze_last_n=50):
    """Create a transfer learning model using MobileNetV2."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Concatenate()([inputs, inputs, inputs])  # replicate grayscale to 3 channels
    x = augmentation(x) # Apply augmentation
    base_model = get_cached_base_model(input_shape) # Use cached base model
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x) # Global average pooling
    x = layers.Dropout(0.2)(x) # Dropout for regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.base_model = base_model
    return model
