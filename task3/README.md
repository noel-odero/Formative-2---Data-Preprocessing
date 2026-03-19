# Task 3: Audio Data Collection and Processing

## Overview

This task collects voice recordings from all four group members and builds a voiceprint verification model. Each member recorded two audio clips (~30 seconds each).

The pipeline covers:
1. Organising and verifying audio files
2. Visualising waveforms and spectrograms
3. Applying audio augmentations
4. Extracting features (MFCCs, spectral roll-off, energy) into `audio_features.csv`
5. Training two voiceprint models: Random Forest and CNN

## Folder Structure

```
audio/
  member1_noel/
    audio1.ogg
    audio2.ogg
  member2_david/
    audio1.ogg
    audio2.ogg
  member3_modestine/
    audio1.ogg
    audio2.ogg
  member4_henriette/
    audio1.ogg
    audio2.ogg
```

## Steps

### 1. Setting Up the Folder Structure
The folder structure is created programmatically to ensure the notebook is fully reproducible. Each member has a dedicated folder containing their two recordings.

### 2. Audio Collection
Each group member recorded two audio clips (~30 seconds each) and placed them in their respective folders. All 8 files are verified before processing begins.

### 3. Loading and Visualising Audio Samples
For each member and phrase, two plots are displayed side by side:

| Plot | Description |
|---|---|
| Waveform | Shows amplitude (loudness) over time — captures speech rhythm and energy patterns unique to each speaker |
| Spectrogram | Shows frequency content over time using a colour map — brighter colours indicate stronger energy at that frequency and time |

These visualisations confirm recordings are clean and usable before any processing.

### 4. Audio Augmentation
Each original recording is augmented to expand the dataset. Augmentations applied per clip:

| Augmentation | Description |
|---|---|
| Time stretching | Speeds up or slows down the audio without changing pitch |
| Pitch shifting | Shifts pitch up or down by a small number of semitones |
| Adding noise | Adds low-level Gaussian noise to simulate real-world conditions |
| Time shifting | Shifts the audio signal forward or backward in time |

### 5. Feature Extraction
The following features are extracted from each audio clip (original + augmented) using **librosa** and saved to `audio_features.csv`:

| Feature | Description |
|---|---|
| MFCCs (40 coefficients) | Mel-Frequency Cepstral Coefficients — compact representation of the spectral envelope |
| Spectral roll-off | Frequency below which a specified percentage of total spectral energy lies |
| RMS energy | Root Mean Square energy — measure of loudness |

### 6. Model 1 — Random Forest Classifier
A Random Forest classifier is trained on the extracted features:

- Labels are encoded with `LabelEncoder`
- Data is split 80/20 train/test
- Evaluated with accuracy, F1 score, and a confusion matrix

The trained model is saved to `models/voiceprint_model.pkl`.

### 7. Model 2 — CNN Voiceprint Model
A Convolutional Neural Network is built using Keras on MFCC spectrograms:

| Layer | Details |
|---|---|
| Conv2D + BatchNorm + MaxPool | 32 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 64 filters, 3×3 kernel |
| Flatten | — |
| Dense + Dropout | 128 units, 0.5 dropout |
| Dense (output) | Softmax, 4 classes (one per member) |

The trained model is saved to `models/voiceprint_cnn.h5`.

## Saved Models

| File | Description |
|---|---|
| `models/voiceprint_model.pkl` | Trained Random Forest model |
| `models/voiceprint_cnn.h5` | Trained CNN model weights |
| `models/voice_label_encoder.pkl` | Fitted `LabelEncoder` for inference |
| `models/voice_label_mapping.json` | Mapping of encoded labels to member names |

## Output

- `audio_features.csv` — extracted feature vectors for all audio clips (original + augmented)

## Dependencies

```
librosa
soundfile
numpy
pandas
matplotlib
scikit-learn
tensorflow / keras
joblib
```

## Notebook

[`Formative2_task3_audio=processing.ipynb`](Formative2_task3_audio=processing.ipynb) — open in Google Colab or Jupyter.
