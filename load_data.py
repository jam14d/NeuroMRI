from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import training_data_generator, validation_data_generator


# Define constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
COLOR_MODE = "grayscale"
CLASS_MODE = "binary"

# Paths to dataset directories
TRAIN_DIR = "/Users/jamieannemortel/BinaryBrainTumorDataset/train"
VAL_DIR = "/Users/jamieannemortel/BinaryBrainTumorDataset/val"
TEST_DIR = "/Users/jamieannemortel/BinaryBrainTumorDataset/test"

# Data generator for training (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.2
)

# Data generator for validation and test (only rescaling)
test_val_datagen = ImageDataGenerator(rescale=1.0/255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    class_mode=CLASS_MODE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = test_val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    class_mode=CLASS_MODE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
    class_mode=CLASS_MODE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Testing

# Fetch one batch from each generator
train_images, train_labels = next(train_generator)
val_images, val_labels = next(val_generator)
test_images, test_labels = next(test_generator)

# Print shapes to confirm everything is loading properly
print("Train batch shape:", train_images.shape, train_labels.shape)
print("Validation batch shape:", val_images.shape, val_labels.shape)
print("Test batch shape:", test_images.shape, test_labels.shape)

# Optional: visualize some samples
import matplotlib.pyplot as plt

def plot_sample_images(images, labels, title, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Label: {int(labels[i])}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_sample_images(train_images, train_labels, "Sample Training Images")
plot_sample_images(val_images, val_labels, "Sample Validation Images")
plot_sample_images(test_images, test_labels, "Sample Test Images")
