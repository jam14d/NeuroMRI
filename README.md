# Brain Tumor MRI Classifier

A deep learning project to classify brain tumors from MRI scans using a convolutional neural network (CNN). 

---

## Project Structure

```
BrainTumorMRI_Dataset/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

---

## Dataset Split

The dataset is organized into three sets for training, validation, and testing:

| Split         | Size (MB) | Items | Percentage |
|---------------|-----------|-------|------------|
| **Train**     | ~116 MB   | 4,576 | ~64%       |
| **Validation**| ~28.2 MB  | 1,146 | ~16%       |
| **Test**      | ~28.8 MB  | 1,316 | ~18%       |
| **Total**     | ~173 MB   | 7,038 | **100%**   |

---

### Clone the repository

```bash
git clone https://github.com/your-username/NeuroMRI-Classifier.git
cd NeuroMRI-Classifier
```

### Create and activate a virtual environment

```bash
python3 -m venv neuro-mri
source neuro-mri/bin/activate  
```

### Install dependencies

```bash
pip install -r requirements.txt
```

