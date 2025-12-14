from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import io
import pickle
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.pkl'
model = None
scaler = None

# Load model and scaler at startup
if os.path.exists(MODEL_PATH):
    try:
        model_data = pickle.load(open(MODEL_PATH, 'rb'))
        
        # Check if model_data is a dict (new format) or just the model (old format)
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            print("Successfully loaded trained model + scaler from model.pkl")
        else:
            model = model_data
            scaler = None
            print("Loaded model (old format, no scaler)")
            
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using mock predictions instead")
else:
    print("Cannot find trained model (model.pkl)")
    print("Using mock predictions for demo")

def extract_features(audio_data, sr=22050):
    """Extract audio features with robust pitch detection - returns features + debug values"""
    try:
        # MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).T, axis=0)
        
        # ROBUST PITCH DETECTION using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sr,
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
            print(f"  Warning: pyin failed ({e}), using fallback")
            # Fallback to piptrack if pyin fails
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_values = pitches[pitches > 0]
            pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
            pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # Spectral features
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr).T, axis=0)
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=audio_data))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
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
        # Return features + pitch_mean + debug values (pitch_std, spectral_centroids, zcr)
        return features, pitch_mean, pitch_std, spectral_centroids, zcr
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0, 0

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        print("Loading audio...")
        audio_data, sr = librosa.load(io.BytesIO(audio_file.read()), sr=22050)
        
        if len(audio_data) < sr * 0.5:
            return jsonify({'error': 'Audio too short. Please record at least 1 second.'}), 400
        
        print(f"Audio loaded: {len(audio_data)/sr:.2f} seconds")
        
        # Extract features
        print("Extracting features...")
        features, pitch_mean, pitch_variance, spectral_centroid, zcr = extract_features(audio_data, sr)
        
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500
        
        print(f"  Pitch Mean: {pitch_mean:.2f} Hz")
        print(f"  Pitch Variance: {pitch_variance:.2f} Hz")
        print(f"  Spectral Centroid: {spectral_centroid:.2f} Hz")
        print(f"  Zero Crossing Rate: {zcr:.4f}")
        
        # Make prediction
        if model is not None:
            print("Using trained model...")
            
            # Scale features if scaler is available
            if scaler is not None:
                features_scaled = scaler.transform([features])
                print("  Features scaled")
            else:
                features_scaled = [features]
                print("  No scaler (using raw features)")
            
            # Predict
            prediction_num = int(model.predict(features_scaled)[0])
            prediction = "singing" if prediction_num == 1 else "talking"
            
            # Get probability
            probas = model.predict_proba(features_scaled)[0]
            confidence = float(probas.max())
            
            print(f"  Model Probabilities: Talking={probas[0]*100:.1f}%, Singing={probas[1]*100:.1f}%")
            print(f"  Model Prediction: {prediction} ({confidence*100:.1f}% confidence)")
            
            # ============================================
            # RULE-BASED OVERRIDE (Fix misclassifications)
            # ============================================
            
            original_prediction = prediction
            original_confidence = confidence
            
            # Check if features strongly indicate TALKING
            is_likely_talking = (
                10 <= pitch_variance <= 100 and  # Talking pitch variance: 20-100 Hz
                85 <= pitch_mean <= 300 and      # Talking pitch range (including high voices)
                0.08 <= zcr <= 0.20              # Talking ZCR (relaxed range)
            )
            
            # Check if features strongly indicate SINGING
            is_likely_singing = (
                200 <= pitch_mean <= 1000 and    # Singing pitch range: 200-1000 Hz
                5 <= pitch_variance <= 60 and    # Stable pitch (singing): 5-60 Hz
                spectral_centroid >= 1500 and    # Bright sound (harmonics)
                0.03 <= zcr <= 0.12              # Low ZCR (sustained phonation)
            )
            
            # Override if features strongly disagree with model
            if is_likely_talking and prediction == "singing":
                print(f"  🔧 OVERRIDE: Features strongly indicate TALKING")
                print(f"     - Pitch variance ({pitch_variance:.1f}) in talking range (10-100 Hz)")
                print(f"     - Pitch mean ({pitch_mean:.1f}) in talking range (85-300 Hz)")
                print(f"     - ZCR ({zcr:.3f}) matches talking pattern")
                prediction = "talking"
                confidence = 0.82  # High confidence in override
                
            elif is_likely_singing and prediction == "talking":
                print(f"  🔧 OVERRIDE: Features strongly indicate SINGING")
                print(f"     - Pitch mean ({pitch_mean:.1f}) in singing range (200-1000 Hz)")
                print(f"     - Pitch variance ({pitch_variance:.1f}) is stable (5-60 Hz)")
                print(f"     - Spectral centroid ({spectral_centroid:.1f}) is bright (>1500 Hz)")
                prediction = "singing"
                confidence = 0.82  # High confidence in override
            
            # Adjust confidence based on feature consistency
            else:
                # Calculate feature match score
                score = 0
                total = 0
                
                if prediction == "talking":
                    # Check talking features
                    if 10 <= pitch_variance <= 100:
                        score += 1
                    total += 1
                    
                    if 85 <= pitch_mean <= 300:
                        score += 1
                    total += 1
                    
                    if 0.08 <= zcr <= 0.20:
                        score += 1
                    total += 1
                    
                else:  # singing
                    # Check singing features
                    if 200 <= pitch_mean <= 1000:
                        score += 1
                    total += 1
                    
                    if 5 <= pitch_variance <= 60:
                        score += 1
                    total += 1
                    
                    if spectral_centroid >= 1500:
                        score += 1
                    total += 1
                
                match_percentage = score / total if total > 0 else 0
                
                # Boost confidence if features match well
                if match_percentage >= 0.8:
                    confidence = max(confidence, 0.80)  # Boost to at least 80%
                elif match_percentage >= 0.6:
                    confidence = max(confidence, 0.70)  # Boost to at least 70%
                # Keep model confidence if features don't match well
            
            if prediction != original_prediction:
                print(f"  ✅ FINAL (after override): {prediction} ({confidence*100:.1f}% confidence)")
            else:
                print(f"  ✅ FINAL: {prediction} ({confidence*100:.1f}% confidence)")
            
        else:
            print("Using mock prediction (no trained model)...")
            
            # Simple heuristic
            if pitch_variance > 200 and spectral_centroid > 2000:
                prediction = "singing"
                confidence = 0.70 + np.random.random() * 0.25
            else:
                prediction = "talking"
                confidence = 0.70 + np.random.random() * 0.25
            
            confidence = float(confidence)
            print(f"  Mock result: {prediction} ({confidence*100:.1f}%)")
        
        # Prepare response
        response = {
            'prediction': str(prediction),
            'confidence': float(confidence),
            'features': {
                'pitch_mean': float(pitch_mean),
                'pitch_variance': float(pitch_variance),
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zcr)
            },
            'using_trained_model': bool(model is not None)
        }
        
        print(f"Classification complete\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'message': 'Backend is running successfully!'
    })

@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        'name': 'Singing vs Talking Classifier API',
        'version': '2.1',
        'endpoints': {
            '/classify': 'POST - Upload audio for classification',
            '/health': 'GET - Check server health',
            '/': 'GET - API information'
        },
        'model_status': {
            'model': 'trained model loaded' if model else 'using mock predictions',
            'scaler': 'feature scaler loaded' if scaler else 'no scaler'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Singing vs Talking Classifier Backend v2.1")
    print("="*50)
    print(f"Model: {'Loaded' if model else 'Not loaded'}")
    print(f"Scaler: {'Loaded' if scaler else 'Not loaded'}")
    print("Server: http://localhost:5000")
    print("Features: Rule-based override enabled")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')