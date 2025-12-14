"""
XGBoost Training Script - Upgraded from Random Forest
Simple drop-in replacement with better performance
"""

import os
import librosa
import numpy as np
from xgboost import XGBClassifier  # ← Changed from RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  # ← Added for better performance
import pickle
import glob
from tqdm import tqdm

DATASET_PATH = 'dataset/'
SINGING_PATH = os.path.join(DATASET_PATH, 'singing/')
TALKING_PATH = os.path.join(DATASET_PATH, 'talking/')
MODEL_PATH = 'model.pkl'
SAMPLE_RATE = 22050
RANDOM_STATE = 42

def extract_features(file_path, sr=22050):
    """Extract audio features with robust pitch detection"""
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=sr, duration=30)
        
        # MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).T, axis=0)
        
        # ROBUST PITCH DETECTION using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sample_rate,
                frame_length=2048
            )
            
            # Filter out unvoiced frames and NaN values
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                pitch_mean = float(np.mean(f0_clean))
                pitch_std = float(np.std(f0_clean))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
                
        except Exception as e:
            print(f"  Warning: pyin failed ({e}), using fallback pitch detection")
            # Fallback to piptrack if pyin fails
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = pitches[pitches > 0]
            pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
            pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # Spectral features
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T, axis=0)
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=audio_data))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        except:
            tempo = 120.0
    
        # Concatenate all features
        features = np.concatenate([
            mfccs,                    # 13 features
            [pitch_mean, pitch_std],  # 2 features
            [spectral_centroids, spectral_rolloff, spectral_bandwidth],  # 3 features
            [zcr, rms, tempo],        # 3 features
            chroma                    # 12 features
        ])
        
        # Total: 13 + 2 + 3 + 3 + 12 = 33 features
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_dataset():
    """Load all audio files and extract features"""
    print("Dataset currently loading...")
    print("_"*30)
    
    X = []
    y = []
    
    singing_files = glob.glob(os.path.join(SINGING_PATH, '*.wav'))
    print(f"\nFound {len(singing_files)} singing samples")
    print("Extracting features from singing samples...")
    
    for file_path in tqdm(singing_files):
        features = extract_features(file_path, SAMPLE_RATE)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 = singing
    
    talking_files = glob.glob(os.path.join(TALKING_PATH, '*.wav'))
    print(f"\nFound {len(talking_files)} talking samples")
    print("Extracting features from talking samples...")
    
    for file_path in tqdm(talking_files):
        features = extract_features(file_path, SAMPLE_RATE)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 = talking
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset Summary:")
    print(f"  _ Total samples: {len(X)}")
    print(f"  _ Singing: {np.sum(y == 1)} samples")
    print(f"  _ Talking: {np.sum(y == 0)} samples")
    print(f"  _ Features per sample: {X.shape[1]}")
    
    return X, y

def train_model(X, y):
    """Train XGBoost classifier"""
    print("Training Model")
    print("_"*30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"  _ Train set: {len(X_train)} samples")
    print(f"  _ Test set: {len(X_test)} samples")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    print("\nTraining classifier...")
    print("  _ Estimators: 200")
    print("  _ Max depth: 6")
    print("  _ Learning rate: 0.1")
    
    model = XGBClassifier(
        n_estimators=200,           # Number of boosting rounds
        max_depth=6,                # Maximum tree depth
        learning_rate=0.1,          # Learning rate
        subsample=0.8,              # Fraction of samples per tree
        colsample_bytree=0.8,       # Fraction of features per tree
        gamma=0,                    # Minimum loss reduction
        reg_alpha=0,                # L1 regularization
        reg_lambda=1,               # L2 regularization
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0                 # Suppress warnings
    )
    
    model.fit(X_train_scaled, y_train)
    print("Training complete!")
    
    # Evaluation
    print("Evaluating model...")
    print("_"*30)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Talking', 'Singing'],
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print("                Predicted")
    print("               Talk    Sing")
    print(f"Actual Talk    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Sing    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled_full, y, cv=5, scoring='accuracy')
    print(f"  CV Scores: {cv_scores}")
    print(f"  Average: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_names = [
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13',
        'Pitch_Mean', 'Pitch_Std',
        'Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
        'ZCR', 'RMS', 'Tempo'
    ] + [f'Chroma_{i+1}' for i in range(12)]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")
    
    # Confidence distribution
    print("\nConfidence Distribution on Test Set:")
    max_probs = y_pred_proba.max(axis=1)
    high_conf = np.sum(max_probs >= 0.9)
    med_conf = np.sum((max_probs >= 0.75) & (max_probs < 0.9))
    low_conf = np.sum(max_probs < 0.75)
    
    print(f"  _ High confidence (≥90%): {high_conf}/{len(y_test)} ({high_conf/len(y_test)*100:.1f}%)")
    print(f"  _ Medium confidence (75-89%): {med_conf}/{len(y_test)} ({med_conf/len(y_test)*100:.1f}%)")
    print(f"  _ Low confidence (<75%): {low_conf}/{len(y_test)} ({low_conf/len(y_test)*100:.1f}%)")
    
    return model, scaler, accuracy

def save_model(model, scaler, path):

    print("\nSAVING MODEL")
    print("_"*30)
    
    # Save both model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_type': 'xgboost'
    }
    
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {path}")
    file_size = os.path.getsize(path) / 1024
    print(f"  File size: {file_size:.2f} KB")

def main():
    print("Singing/talking Classifier model")
    print("_"*30)
    
    # Check dataset
    if not os.path.exists(SINGING_PATH) or not os.path.exists(TALKING_PATH):
        print("\nERROR: Dataset folders not found!")
        print(f"  Please create:")
        print(f"    _ {SINGING_PATH}")
        print(f"    _ {TALKING_PATH}")
        print("  And add .wav audio files")
        return
    
    singing_count = len(glob.glob(os.path.join(SINGING_PATH, '*.wav')))
    talking_count = len(glob.glob(os.path.join(TALKING_PATH, '*.wav')))
    
    if singing_count == 0 or talking_count == 0:
        print("\nERROR: No audio files found!")
        print(f"  Singing files: {singing_count}")
        print(f"  Talking files: {talking_count}")
        return
    
    # Load data
    X, y = load_dataset()
    
    if len(X) < 10:
        print("\nDataset is very small!")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Train
    model, scaler, accuracy = train_model(X, y)
    
    # Save
    save_model(model, scaler, MODEL_PATH)
    
    print("\nTraining complete!")
    
    print(f"\nModel: Classifier")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Saved to: {MODEL_PATH}" + "\n")
    
if __name__ == '__main__':
    main()