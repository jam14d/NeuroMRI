import os
import shutil
from pathlib import Path

"""
Copy data into binary dataset.
"""

# Path to your original dataset
original_dataset = Path("/Users/jamieannemortel/BrainTumorMRI_Dataset")
binary_dataset = original_dataset.parent / "BinaryBrainTumorDataset"

# Classes to keep
selected_classes = ["glioma", "no_tumor"]

# Create directory structure for new dataset
for split in ["train", "val", "test"]:
    for class_name in selected_classes:
        target_dir = binary_dataset / split / class_name
        source_dir = original_dataset / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in source_dir.glob("*"):
            shutil.copy(file, target_dir / file.name)

print(f"Binary dataset created at: {binary_dataset}")
