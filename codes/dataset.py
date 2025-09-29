from keras import layers
import tensorflow as tf

def create_dataset_from_directory(directory, image_size=(513, 450), batch_size=2, shuffle=True):
    """Creates a TensorFlow dataset from a directory of images."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        color_mode='grayscale',
    )
    normalization_layer = layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))
