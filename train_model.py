from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from load_data import get_data_generators
import matplotlib.pyplot as plt
from datetime import datetime
import os

# CONFIGURATION
DEBUG_MODE = False
EPOCHS = 1 if DEBUG_MODE else 15
STEPS_PER_EPOCH = 2 if DEBUG_MODE else None
VAL_STEPS = 1 if DEBUG_MODE else None
IMAGE_SHAPE = (256, 256, 1)

# Timestamped model save path
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = '/content/drive/MyDrive/Models'
os.makedirs(model_dir, exist_ok=True)
MODEL_PATH = os.path.join(model_dir, f'glioma_classifier_{timestamp}.h5')

# LOAD DATA
train_generator, val_generator, test_generator = get_data_generators(debug_mode=DEBUG_MODE)

# MODEL DEFINITION (deeper CNN with BatchNormalization)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SHAPE),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS
callbacks = []
if not DEBUG_MODE:
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor='val_loss')
    callbacks = [early_stopping, checkpoint]

# TRAINING
print("Starting training..." + (" (DEBUG MODE)" if DEBUG_MODE else ""))
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# FINAL MODEL SAVE
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# EVALUATE
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc:.2%}")

# PLOT TRAINING HISTORY
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
