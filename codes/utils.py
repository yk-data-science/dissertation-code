import os
import sys
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TeeLogger(object):
    """Custom logger that writes output to both the console and a log file."""
    def __init__(self, file_path):
        self.terminal = sys.stdout # Standard output
        self.log = open(file_path, "w", encoding="utf-8")
    def write(self, message): # Write to both terminal and log file
        self.terminal.write(message)
        self.log.write(message)
    def flush(self): # Flush both outputs
        self.terminal.flush()
        self.log.flush()

def list_image_filenames(directory):
    """Return a list of image filenames in the directory"""
    return [
        file for root, _, files in os.walk(directory)
        for file in files
        if file.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

def setup_tensorflow_threads(num_threads):
    """Set TensorFlow threading configuration."""
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
