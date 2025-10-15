import os
from datetime import datetime

seed = 42 # For reproducibility
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
condition = "with_aug_seed" # or "without_aug_seed"
target = "Treatment"  # or "VA"
base_dir = "../data/processed/treatment_split_into_two_groups_14"
# or
# base_dir = "../data/processed/va_split_into_two_groups_70"
input_shape = (450, 450, 1) # (height, width, channels)
patience = 7 # Early stopping patience
epochs = 200 # Maximum number of epochs

# Cross-validation splits
splits = ['split1', 'split2', 'split3', 'split4', 'split5', 'split6', 'split7']

# Parallel execution settings
max_workers = min(len(splits), os.cpu_count())

# Augmentation: (flip, zoom, contrast, noise)
augmentation_configs = [
    ("horizontal", 0.2, 0.3, 0.05),
    ("horizontal", 0.3, 0.2, 0.03),
    ("horizontal", 0.1, 0.4, 0.05),
    ("horizontal", 0.2, 0.2, 0.02),
    ("horizontal", 0.1, 0.4, 0.05),
]

model_name = "create_cnn_for_oct_biomarker"
# model_name =  "create_transfer_model"
num_classes = 2 # Binary classification ... this time only

save_dir = f"../outputs/{target}/" # output directory
os.makedirs(save_dir, exist_ok=True) # make directory if not exists
