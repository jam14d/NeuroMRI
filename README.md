# NeuroMRI

This is a deep learning project aiming to automatically classify brain MRI scans as either showing signs of a **glioma tumor** or indicating **no tumor**. 


![Glioma - GradCAM](images/TP_GRADCAM.webp)
**Example output from gradcam.py**.
The model correctly predicts “Glioma” with 100% confidence. Grad-CAM highlights central regions correlated with tumor presence.



## How to Run

You can run everything in Google Colab, which provides free access to a GPU and a Python environment.

For local -- create and activate a virtual environment, then install the required dependencies:

```bash
python3.9 -m venv ~/venv-metal
source ~/venv-metal/bin/activate 
```

### 1. **Create Binary Dataset**

Extract only `glioma` and `no_tumor` classes into a new dataset folder.

```bash
python binary_directory.py
```

This will create a `BinaryBrainTumorDataset` folder next to your original dataset.

### 2. **Train the Model**

Train a CNN model using the binary dataset.

```bash
python train_model.py
```

By default, it trains for 15 epochs. You can enable debug mode for quick testing by setting `DEBUG_MODE = True` inside the script.

### 3. **Visualize Model with Grad-CAM**

Generate Grad-CAM visualizations on test images.

```bash
python gradcam.py
```

This loads the saved model and shows Grad-CAM overlays for several test examples.

---

## Dataset Structure

Your dataset should be structured like this before running:

```
BrainTumorMRI_Dataset/
├── train/
│   ├── glioma/
│   ├── no_tumor/
│   └── ... (other classes, ignored)
├── val/
│   ├── glioma/
│   ├── no_tumor/
├── test/
│   ├── glioma/
│   ├── no_tumor/
```

Only `glioma` and `no_tumor` classes will be used in training.

---

## Notes

* The trained model is saved as `glioma_classifier.h5`.
* Grad-CAM visualizations help verify the model focuses on relevant areas in the MRI images.
* `DEBUG_MODE` in `train_model.py` allows fast training/testing for development.

---
The dataset used for this project comes from Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

