import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    https://keras.io/examples/vision/grad_cam/
    """
    last_conv_layer = next( # Find the last convolutional
        layer for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D) # Only consider Conv2D layers
    )
    last_conv_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:]) # Create a model for the classifier part
    x = tf.keras.layers.GlobalAveragePooling2D()(classifier_input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(model.output_shape[-1], activation='softmax')(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_model(img_array, training=False) # Forward pass through the conv layers
        tape.watch(conv_outputs) # Watch the conv layer outputs
        preds = classifier_model(conv_outputs) # Forward pass through the classifier
        if pred_index is None:
            pred_index = tf.argmax(preds[0]) # Use the top predicted class if none specified
        class_channel = preds[:, pred_index] # Get the score for the target class

    grads = tape.gradient(class_channel, conv_outputs) # Compute gradients of the class score w.r.t. conv outputs
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)) # Global average pooling of gradients

    conv_outputs = conv_outputs[0] # Remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # Weighted combination of conv outputs
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-10) # Normalize to [0, 1]

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4, output_path=None):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) # Resize heatmap to image size
    heatmap = np.uint8(255 * heatmap) # Convert to uint8
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Apply colormap
    if img.shape[-1] == 1: # If grayscale, convert to RGB
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) # Convert to 3-channel
    else:
        img_rgb = (img * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0) # Superimpose heatmap on image
    if output_path:
        success = cv2.imwrite(output_path, superimposed_img)
        if not success:
            print(f"⚠️ Failed to save Grad-CAM image to {output_path}")
    return superimposed_img
