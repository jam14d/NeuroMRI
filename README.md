# NeuroMRI

**NeuroMRI** is a deep learning project aiming to automatically classify brain MRI scans as either showing signs of a **glioma tumor** or indicating **no tumor**. This binary classification task aims to support early and accurate detection of brain tumors using medical imaging.

The dataset used for this project comes from Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


## Setup

Create and activate a virtual environment, then install the required dependencies:

```bash
python3.9 -m venv ~/venv-metal
source ~/venv-metal/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
