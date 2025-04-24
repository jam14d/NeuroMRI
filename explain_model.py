import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model
from load_data import get_data_generators

# Load model
model = load_model("glioma_classifier.h5")
model.summary()  # Optional: see the layers

# Load a test image
_, _, test_generator = get_data_generators(test_mode=True)
img_batch, label_batch = next(test_generator)
img = img_batch[0:1]  # (1, 256, 256, 1)
label = int(label_batch[0])  # Ground truth: 0 (no tumor) or 1 (glioma)

# Make prediction
pred = model.predict(img)[0][0]  # raw sigmoid score (0.0 - 1.0)
pred_label = int(pred > 0.5)
confidence = pred if pred_label == 1 else 1 - pred


# 1. Visualize Filters (First Conv Layer)
first_conv = next(layer for layer in model.layers if 'conv' in layer.name)
filters, _ = first_conv.get_weights()

def plot_filters(filters, n=8):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        f = filters[:, :, 0, i]  # assuming grayscale input
        f = (f - f.min()) / (f.max() - f.min())  # normalize
        plt.subplot(1, n, i + 1)
        plt.imshow(f, cmap='gray')
        plt.title(f'Filter {i}')
        plt.axis('off')
    plt.suptitle("First Conv Layer Filters")
    plt.show()

plot_filters(filters)

# 2. Visualize Feature Maps 
activation_model = Model(inputs=model.input, outputs=first_conv.output)
feature_maps = activation_model.predict(img)

def plot_feature_maps_with_prediction(feature_maps, true_label, pred_label, confidence, n=8):
    correct = (true_label == pred_label)
    title_color = "green" if correct else "red"
    label_map = {0: "No Tumor", 1: "Glioma"}

    plt.figure(figsize=(15, 6))

    # Big title
    plt.suptitle(
        f"Prediction: {label_map[pred_label]} ({confidence:.2%} confidence) — "
        f"Ground Truth: {label_map[true_label]}",
        fontsize=14,
        color=title_color
    )

    # Feature maps
    for i in range(n):
        fmap = feature_maps[0, :, :, i]
        plt.subplot(2, n//2, i + 1)
        plt.imshow(fmap, cmap='viridis')
        plt.title(f'Map {i}')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    plt.show()


plot_feature_maps_with_prediction(feature_maps, label, pred_label, confidence)

