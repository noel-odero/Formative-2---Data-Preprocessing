# Formative 2 — Data Preprocessing

A multi-modal machine learning pipeline that combines **tabular data**, **facial images**, and **voice recordings** to build a user identity verification and product recommendation system.

## Project Overview

The system works in three stages:

1. **Task 1** — consolidate customer transaction and social profile datasets, conduct exploratory data analysis (EDA), and develop a predictive model for product categories.
2. **Task 2** — gather and augment facial image data to train a robust biometric recognition system utilizing CNNs and DeepFace..
3. **Task 3** — Collect voice recordings, extract audio features, and build a voiceprint verification model (Random Forest + CNN).

All three models are combined in `simulation.py`, which runs a full end-to-end identity verification and product recommendation pipeline.

---

## Repository Structure

```
Formative-2---Data-Preprocessing/
│
├── Task1/
│   ├── Formative2_task1_Data_merge.ipynb   # Data merge, EDA & product prediction
│   ├── merged_customer_dataset.csv          # Output: cleaned merged dataset
│   └── README.md
│
├── Task2/
│   ├── Task2_image_processing.ipynb         # Face image processing & CNN
│   └── README.md
│
├── task3/
│   ├── Formative2_task3_audio=processing.ipynb  # Audio processing & voiceprint models
│   └── README.md
│
├── faces/                                   # Raw face images (4 members × 3 expressions)
│   ├── member1_noel/
│   ├── member2_david/
│   ├── member3_modestine/
│   └── member4_henriette/
│
├── audio/                                   # Raw voice recordings (4 members × 2 clips)
│   ├── member1_noel/
│   ├── member2_david/
│   ├── member3_modestine/
│   └── member4_henriette/
│
├── models/                                  # All saved model files
│   ├── face_recognition_cnn.h5
│   ├── face_label_mapping.json
│   ├── label_encoder.pkl
│   ├── product_recommendation_model.pkl
│   ├── voice_label_encoder.pkl
│   ├── voice_label_mapping.json
│   ├── voiceprint_cnn.h5
│   └── voiceprint_model.pkl
│
├── test_inputs/                             # Sample inputs for the simulation
│   ├── authorized_face.jpeg
│   ├── authorized_voice.ogg
│   ├── unauthorized_face.jpeg
│   └── unauthorized_voice.ogg
│
├── simulation.py                            # End-to-end simulation script
└── README.md                                # This file
```

---

## Tasks at a Glance

| Task | Input | Models | Output |
|---|---|---|---|
| Task 1 | Customer transactions + social profiles | Logistic Regression, Random Forest, XGBoost | Product category prediction |
| Task 2 | Facial images (3 expressions × 4 members) | CNN (scratch) + DeepFace (VGG-Face) | Face identity verification |
| Task 3 | Voice recordings (2 clips × 4 members) | Random Forest + CNN | Voiceprint verification |

---

## End-to-End Simulation (`simulation.py`)

The simulation script ties all three tasks together into a single pipeline:

```
Step 1 → Face verification   (DeepFace VGG-Face)
Step 2 → Voice verification  (Random Forest voiceprint model)
Step 3 → Product recommendation (Random Forest trained on merged dataset)
```

### Usage

**Authorised attempt:**
```bash
python simulation.py \
  --face  test_inputs/authorized_face.jpeg \
  --voice test_inputs/authorized_voice.ogg \
  --member member1_noel
```

**Unauthorised attempt:**
```bash
python simulation.py \
  --face  test_inputs/unauthorized_face.jpeg \
  --voice test_inputs/unauthorized_voice.ogg \
  --member member1_noel
```

### Arguments

| Argument | Description | Example |
|---|---|---|
| `--face` | Path to the probe face image | `test_inputs/authorized_face.jpeg` |
| `--voice` | Path to the probe voice recording | `test_inputs/authorized_voice.ogg` |
| `--member` | Claimed member identity | `member1_noel` |

### Flow

1. **Face verification** — DeepFace compares the probe image against the member's registered `neutral.jpeg`. If the distance exceeds the threshold (0.40), access is denied immediately.
2. **Voice verification** — 42 audio features (40 MFCCs + spectral roll-off + RMS energy) are extracted and passed to the Random Forest voiceprint model. If the predicted speaker does not match the claimed member, access is denied.
3. **Product recommendation** — If both checks pass, the product recommendation model predicts the most likely product category for the verified user.

---

## Group Members

| Member | Folder |
|---|---|
| Noel | `member1_noel` |
| David | `member2_david` |
| Modestine | `member3_modestine` |
| Henriette | `member4_henriette` |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow / keras
opencv-python
Pillow
librosa
soundfile
deepface
joblib
openpyxl
```

---



## Task READMEs

- [Task 1 README](Task1/README.md) — Data merge, EDA & product prediction
- [Task 2 README](Task2/README.md) — Image processing & face recognition
- [Task 3 README](task3/README.md) — Audio processing & voiceprint verification
