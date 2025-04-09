from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Define dataset paths
base_dir = "/Users/jamieannemortel/BrainTumorMRI_Dataset"  
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05
)
