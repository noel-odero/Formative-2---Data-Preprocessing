#!/usr/bin/env python3
"""
Flow:
1. Face verification (DeepFace) → Access granted/denied
2. Voice verification (Random Forest) → Transaction approved/denied
3. Product recommendation (Random Forest) → Product predicted

Usage:
    # Authorized attempt:
    python simulate.py  --face test_inputs/authorized_face.jpeg \
                        --voice test_inputs/authorized_voice.ogg \
                        --member member1_noel

    # Unauthorized attempt:
    python simulate.py  --face test_inputs/unauthorized_face.jpeg \
                        --voice test_inputs/unauthorized_voice.ogg \
                        --member member1_noel
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import librosa
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder as LE

# PATHS
MODELS_DIR  = 'models'
FACES_DIR   = 'faces'

PRODUCT_MODEL_PATH   = os.path.join(MODELS_DIR, 'product_recommendation_model.pkl')
LABEL_ENCODER_PATH   = os.path.join(MODELS_DIR, 'label_encoder.pkl')
VOICE_MODEL_PATH     = os.path.join(MODELS_DIR, 'voiceprint_model.pkl')
VOICE_ENCODER_PATH   = os.path.join(MODELS_DIR, 'voice_label_encoder.pkl')

# HELPERS

def status(message):
    print(f"    {message}")

def extract_audio_features(y, sr):
    """Extract 42 features: 40 MFCCs + spectral roll-off + RMS energy."""
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean  = np.mean(mfccs, axis=1)
    rolloff    = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms        = np.mean(librosa.feature.rms(y=y))
    return np.concatenate([mfcc_mean, [rolloff, rms]]).reshape(1, -1)

def loading(message, duration=1.2):
    """Simple loading animation."""
    print(f"\n  {message}", end='', flush=True)
    for _ in range(3):
        time.sleep(duration / 3)
        print('.', end='', flush=True)
    print()

# LOAD MODELS
def load_models():
    try:
        product_model  = joblib.load(PRODUCT_MODEL_PATH)
        label_encoder  = joblib.load(LABEL_ENCODER_PATH)
        voice_model    = joblib.load(VOICE_MODEL_PATH)
        voice_encoder  = joblib.load(VOICE_ENCODER_PATH)

        return product_model, label_encoder, voice_model, voice_encoder
    except FileNotFoundError as e:
        print(f"\n Model file not found: {e}")
        print("  Make sure all models are in the 'models/' folder.")
        sys.exit(1)

# STEP 1 - FACE VERIFICATION
def verify_face(probe_path, claimed_member):
    """
    Compare probe face against the registered neutral image
    for the claimed member using DeepFace VGG-Face.
    Returns (verified: bool, distance: float)
    """
    ref_path = os.path.join(FACES_DIR, claimed_member, 'neutral.jpeg')

    if not os.path.exists(ref_path):
        print(f"  Reference image not found for {claimed_member}")
        return False, 9.999

    if not os.path.exists(probe_path):
        print(f"  Probe image not found: {probe_path}")
        return False, 9.999

    try:
        result = DeepFace.verify(
            ref_path, probe_path,
            model_name='VGG-Face',
            enforce_detection=False
        )
        return result['verified'], round(result['distance'], 4)
    except Exception as e:
        print(f"  DeepFace error: {e}")
        return False, 9.999

# STEP 2 - VOICE VERIFICATION
def verify_voice(audio_path, claimed_member, voice_model, voice_encoder):
    """
    Extract features from probe audio and predict speaker identity.
    Returns (verified: bool, predicted_member: str, confidence: float)
    """
    if not os.path.exists(audio_path):
        print(f"  Audio file not found: {audio_path}")
        return False, 'unknown', 0.0

    try:
        y, sr   = librosa.load(audio_path, sr=22050)
        features = extract_audio_features(y, sr)
        pred_idx = voice_model.predict(features)[0]
        proba    = voice_model.predict_proba(features)[0]
        confidence = round(proba[pred_idx] * 100, 2)
        predicted_member = voice_encoder.inverse_transform([pred_idx])[0]
        verified = (predicted_member == claimed_member)
        return verified, predicted_member, confidence
    except Exception as e:
        print(f"  Voice verification error: {e}")
        return False, 'unknown', 0.0

# STEP 3 - PRODUCT RECOMMENDATION
def get_product_recommendation(product_model, label_encoder):
    """
    Generate a product recommendation for the verified member.
    Uses median feature values as a representative input vector.
    """
    # Load merged dataset to get a representative sample for this simulation
    merged_path = 'Task1/merged_customer_dataset.csv'
    if not os.path.exists(merged_path):
        merged_path = 'merged_customer_dataset.csv'

    try:
        df = pd.read_csv(merged_path)

        # Encode categoricals the same way as training
        cat_cols = ['social_media_platform', 'review_sentiment']
        for col in cat_cols:
            if col in df.columns:
                df[col] = LE().fit_transform(df[col].astype(str))

        # Drop non-feature columns
        drop_cols = ['transaction_id', 'purchase_date', 'customer_id', 'product_category', 'product_category_encoded']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_sample = df[feature_cols].median().values.reshape(1, -1)

        pred_encoded  = product_model.predict(X_sample)[0]
        pred_proba    = product_model.predict_proba(X_sample)[0]
        confidence    = round(max(pred_proba) * 100, 2)
        product       = label_encoder.inverse_transform([pred_encoded])[0]
        return product, confidence

    except Exception as e:
        print(f"  Recommendation error: {e}")
        return 'Unknown', 0.0

# MAIN SIMULATION
def run_simulation(face_path, voice_path, claimed_member):

    # Load all models
    product_model, label_encoder, voice_model, voice_encoder = load_models()

    #  HEADER 
   
    print("   USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")
   
    print(f"  Claimed Identity : {claimed_member.split('_')[1].capitalize()}")
    print(f"  Face Input       : {face_path}")
    print(f"  Voice Input      : {voice_path}")
   

    #  STEP 1: FACE VERIFICATION 
    print("\n  [ STEP 1 ] FACIAL VERIFICATION")
    loading("  Running DeepFace VGG-Face verification")

    face_verified, face_distance = verify_face(face_path, claimed_member)

    if face_verified:
        status( f"Face VERIFIED  |  Distance: {face_distance}  (threshold: 0.40)")
    else:
        status( f"Face NOT verified  |  Distance: {face_distance}  (threshold: 0.40)")
       
        print("\n  ACCESS DENIED - Face verification failed.")
        print("      The system has rejected this access attempt.")
       
        return

    #  STEP 2: VOICE VERIFICATION 
    print("\n  [ STEP 2 ] VOICE VERIFICATION")
    loading("  Analysing voiceprint")

    voice_verified, predicted_member, voice_confidence = verify_voice(
        voice_path, claimed_member, voice_model, voice_encoder
    )

    if voice_verified:
        status( f"Voice VERIFIED  |  Predicted: "
                     f"{predicted_member.split('_')[1].capitalize()}"
                     f"  |  Confidence: {voice_confidence}%")
    else:
        status( f"Voice NOT verified  |  Predicted: "
                     f"{predicted_member.split('_')[1].capitalize()}"
                     f"  |  Confidence: {voice_confidence}%")
       
        print("\n  ACCESS DENIED - Voice verification failed.")
        print("      Both face and voice must be verified to proceed.")
       
        return

    #  STEP 3: PRODUCT RECOMMENDATION 
    print("\n  [ STEP 3 ] PRODUCT RECOMMENDATION")
    loading("  Generating personalised recommendation")

    product, prod_confidence = get_product_recommendation(
        product_model, label_encoder
    )

    status( f"Recommended Product : {product}")
    status( f"Model Confidence    : {prod_confidence}%")

    #  SUMMARY 
   
    print("\n  TRANSACTION COMPLETE")
    print(f"\n  Welcome, {claimed_member.split('_')[1].capitalize()}!")
    print(f"  Based on your profile, we recommend: {product}")
   


# ENTRY POINT
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='User Identity & Product Recommendation System Simulation'
    )
    parser.add_argument('--face',   required=True,
                        help='Path to face image (e.g. test_inputs/authorized_face.jpeg)')
    parser.add_argument('--voice',  required=True,
                        help='Path to voice recording (e.g. test_inputs/authorized_voice.ogg)')
    parser.add_argument('--member', required=True,
                        help='Claimed member identity (e.g. member1_noel)')

    args = parser.parse_args()
    run_simulation(args.face, args.voice, args.member)