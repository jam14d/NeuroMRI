from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data generator with augmentation
training_data_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.2
)

# Validation/Test data generator without augmentation
validation_data_generator = ImageDataGenerator(rescale=1.0/255)
