from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Uses Keras ImageDataGenerator to prepare data for training/testing with augmentation (train only)
"""

# Constants
IMAGE_SIZE = (256, 256)
COLOR_MODE = "grayscale"
CLASS_MODE = "binary"

# Directory paths
TRAIN_DIR = '/content/drive/MyDrive/BinaryBrainTumorDataset/train'
VAL_DIR   = '/content/drive/MyDrive/BinaryBrainTumorDataset/val'
TEST_DIR  = '/content/drive/MyDrive/BinaryBrainTumorDataset/test'



def get_data_generators(debug_mode=False, batch_size=128):
    """
    Returns train, val, and test generators.

    If debug_mode is True:
    - Uses a small batch size (e.g. 4) for fast pipeline testing
    - Intended for sanity checks, not real training or evaluation
    """
    if debug_mode:
        print("DEBUG MODE: Using small batch size for quick pipeline test.")
        batch_size = 4  # Tiny batch for debugging only

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2
        horizontal_flip=True  # added
    )

    # No augmentation for val/test
    test_val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_generator = test_val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=batch_size,
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator, test_generator
