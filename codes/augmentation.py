from keras import layers

def build_augmentation_layer(flip, zoom, contrast, noise):
    """
    Build a configurable augmentation function (flip, zoom, contrast, noise) for training images.
    """
    def augment(inputs):
        """ Apply specified augmentations to the input images."""
        x = inputs
        if flip == "vertical":
            x = layers.RandomFlip("vertical")(x)
        elif flip == "horizontal":
            x = layers.RandomFlip("horizontal")(x)
        elif flip == "horizontal_and_vertical":
            x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomZoom(zoom)(x)
        x = layers.RandomContrast(contrast)(x)
        x = layers.GaussianNoise(noise)(x)
        return x
    return augment
