import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import collections
from tensorflow.keras.models import load_model
from load_data import get_data_generators


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    if grads is None:
        print("⚠️ No gradients returned — possibly wrong layer or flat prediction.")
        return np.zeros_like(conv_outputs[0, :, :, 0].numpy())

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()

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

#  Run on one or more test images 
if __name__ == "__main__":
    model = load_model("glioma_classifier.h5")
    model.summary()

    # Choose the last conv layer manually if needed
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    print("Using last conv layer:", last_conv_layer_name)

    _, _, test_generator = get_data_generators(test_mode=True)

    # Find a glioma (label = 1) test image
    for i in range(10):
        img_batch, label_batch = next(test_generator)
        # Count how many images of each class are in the batch
        int_labels = [int(lbl) for lbl in label_batch]
        counts = collections.Counter(int_labels)
        label_map = {0: "No Tumor", 1: "Glioma"}

        print("\nClass distribution in batch:")
        for class_id in sorted(counts):
            print(f"{label_map[class_id]} ({class_id}): {counts[class_id]}")

        if int(label_batch[0]) == 1:
            img = img_batch[0:1]
            label = 1
            print(f"\nFound a glioma image (Index {i})")
            break
    else:
        print("No glioma image found in test batch. Try running again or using a bigger test batch.")
        exit()

    # Predict and run Grad-CAM
    pred_score = model.predict(img)[0][0]
    pred_label = int(pred_score > 0.5)
    confidence = pred_score if pred_label == 1 else 1 - pred_score
    print(f"Predicted: {pred_label} ({confidence:.2%} confidence) | True Label: {label}")

    # Force Grad-CAM to show glioma class (even if model guessed otherwise)
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=1)

    if np.max(heatmap) < 0.1:
        print("Heatmap seems weak — still no strong gradients.")
    else:
        show_gradcam(img, heatmap, pred_label, confidence, label)


