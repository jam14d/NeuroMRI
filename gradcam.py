import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from load_data import get_data_generators


"""
Loads the trained model and test data to visualize predictions using Grad-CAM heatmaps.

"""

# Make Grad-CAM heatmap (binary classification version)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_output = predictions[:, 0]  # Single scalar output (sigmoid)

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()

# Show Grad-CAM visualization
def show_gradcam(image, heatmap, prediction, confidence, label, alpha=0.4):
    label_map = {0: "No Tumor", 1: "Glioma"}
    correct = int(prediction == label)
    color = "green" if correct else "red"

    image = image.squeeze()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    plt.figure(figsize=(12, 4))
    plt.suptitle(
        f"Prediction: {label_map[prediction]} ({confidence:.1%}) — Ground Truth: {label_map[label]}",
        fontsize=14, color=color
    )

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed[..., ::-1])  # BGR to RGB
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    model = load_model("glioma_classifier.h5")
    model.summary()

    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    print("Using last conv layer:", last_conv_layer_name)

    _, _, test_generator = get_data_generators(debug_mode=False)

    # Buffers
    glioma_samples = []
    no_tumor_samples = []

    # Collect images
    print("Collecting Glioma and No Tumor images...")
    while len(glioma_samples) < 5 or len(no_tumor_samples) < 5:
        img_batch, label_batch = next(test_generator)

        for i in range(len(label_batch)):
            label = int(label_batch[i])
            img = img_batch[i:i+1]

            if label == 1 and len(glioma_samples) < 5:
                glioma_samples.append((img, label))
            elif label == 0 and len(no_tumor_samples) < 5:
                no_tumor_samples.append((img, label))

            if len(glioma_samples) >= 5 and len(no_tumor_samples) >= 5:
                break

    # Combine and shuffle
    samples = glioma_samples + no_tumor_samples
    np.random.shuffle(samples)

    # Run Grad-CAM on all
    for img, label in samples:
        pred_score = model.predict(img)[0][0]
        pred_label = int(pred_score > 0.5)
        confidence = pred_score if pred_label == 1 else 1 - pred_score

        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

        show_gradcam(img, heatmap, pred_label, confidence, label)
