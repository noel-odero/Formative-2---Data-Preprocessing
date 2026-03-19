# Task 2: Image Data Collection and Processing

## Overview

This task collects facial images for each group member, applies augmentations, extracts features, and builds two facial recognition models:

1. A **CNN trained from scratch** using the collected images
2. A **DeepFace pretrained model** for identity verification

## Folder Structure

Images are organised as follows:

```
faces/
  member1_noel/
    neutral.jpeg
    smiling.jpeg
    surprised.jpeg
  member2_david/
    neutral.jpeg
    smiling.jpeg
    surprised.jpeg
  member3_modestine/
    neutral.jpeg
    smiling.jpeg
    surprised.jpeg
  member4_henriette/
    neutral.jpeg
    smiling.jpeg
    surprised.jpeg
```

## Steps

### 1. Setting Up the Folder Structure
The folder structure is created programmatically so the notebook is fully reproducible on any machine. Each member gets a dedicated folder for their three expression images.

### 2. Image Collection
Each group member took 3 photos — one **neutral**, one **smiling**, and one **surprised** — and placed them in their respective folders. All 12 images are verified before processing begins.

### 3. Loading and Displaying Images
All images are loaded with OpenCV, resized to 150×150, and displayed in a 4×3 grid (one row per member, one column per expression) to confirm correct loading.

### 4. Image Augmentation
Each original image is augmented to expand the dataset. Augmentations applied per image:

| Augmentation | Description |
|---|---|
| Horizontal flip | Mirror image left-to-right |
| Rotation (±15°) | Slight clockwise / counter-clockwise rotation |
| Brightness adjustment | Increase and decrease brightness |
| Zoom | Slight zoom-in crop |

This produces a larger, more varied training set from the 12 original images.

### 5. Feature Extraction
Images are preprocessed and flattened into feature vectors for use with the CNN model.

### 6. Model 1 — CNN Trained from Scratch

A Convolutional Neural Network is built using Keras with the following architecture:

| Layer | Details |
|---|---|
| Conv2D + BatchNorm + MaxPool | 32 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 64 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 128 filters, 3×3 kernel |
| Flatten | — |
| Dense + Dropout | 256 units, 0.5 dropout |
| Dense (output) | Softmax, 4 classes (one per member) |

- Labels are encoded with `LabelEncoder` and one-hot encoded with `to_categorical`
- Data is split 80/20 train/test
- Model is evaluated with accuracy and a confusion matrix

The trained model is saved to `models/face_recognition_cnn.h5`.

### 7. Model 2 — DeepFace Identity Verification

The **DeepFace** library is used for identity verification. Given a test image, DeepFace compares it against the stored member images and returns whether the face is **verified** (authorised) or **not verified** (unauthorised).

- Authorised test: `test_inputs/authorized_face.jpeg`
- Unauthorised test: `test_inputs/unauthorized_face.jpeg`

## Saved Models

| File | Description |
|---|---|
| `models/face_recognition_cnn.h5` | Trained CNN model weights |
| `models/face_label_mapping.json` | Mapping of encoded labels to member names |
| `models/label_encoder.pkl` | Fitted `LabelEncoder` for inference |

## Dependencies

```
opencv-python
numpy
matplotlib
Pillow
tensorflow / keras
deepface
scikit-learn
```

## Notebook

[`Task2_image_processing.ipynb`](Task2_image_processing.ipynb) — open in Google Colab or Jupyter.
