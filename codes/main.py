import os
import sys
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

from codes.utils import set_seeds, TeeLogger, setup_tensorflow_threads
from augmentation import build_augmentation_layer
from codes.models import (
    create_simple_cnn,
    create_cnn_for_oct_biomarker,
    create_transfer_model
)
from codes.dataset import create_dataset_from_directory
from codes.config import (
    seed, condition, input_shape, patience, epochs, target,
    augmentation_configs, save_dir, base_dir, timestamp, model_name,
    num_classes, splits
)

MODEL_FACTORY = {
    "create_simple_cnn": create_simple_cnn,
    "create_cnn_for_oct_biomarker": create_cnn_for_oct_biomarker,
    "create_transfer_model": create_transfer_model,
}


def run_split(split):
    """Train and evaluate the model for a given split."""
    # Reproducibility
    set_seeds(seed)
    setup_tensorflow_threads(1) # Limit TF to 1 thread per process

    # Logging
    log_path = f"{save_dir}/logs/{condition}/cnn_{split}_log_{timestamp}.txt"
    sys.stdout = TeeLogger(log_path) # Log to file and console

    # Build augmentation
    aug_config = augmentation_configs[0]  # Choose the first config
    augmentation = build_augmentation_layer(*aug_config)

    # Create model
    model_fn = MODEL_FACTORY[model_name]
    model = model_fn(input_shape, num_classes=num_classes, augmentation=augmentation) # Add augmentation layer
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # Use categorical_crossentropy for 2classes
    model.summary()

    # Datasets
    train_dir = os.path.join(base_dir, split, "train")
    test_dir = os.path.join(base_dir, split, "test")
    train_ds = create_dataset_from_directory(train_dir, image_size=input_shape[:2], batch_size=2)
    test_ds = create_dataset_from_directory(test_dir, image_size=input_shape[:2], batch_size=2)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/{target}/{condition}/best_model_{split}.h5", save_best_only=True),
    ]

    # Train
    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks)

    return {"split": split, "history": history.history}


def main():
    """Run training across all predefined splits in parallel."""
    max_workers = min(len(splits), os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(run_split, splits):
            print(f"Finished training for {result['split']}")


if __name__ == "__main__":
    main()
