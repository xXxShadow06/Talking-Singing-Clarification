Singing vs Talking Classifier

A real-time audio classification web app that uses machine learning to detect whether someone is singing or talking. Features live recording with optional video preview and displays confidence scores with audio feature analysis.

Features

- **Live Audio/Video Recording**: Record with microphone only or enable camera
- **Real-time Classification**: Analyzes audio to detect singing vs talking
- **Feature Visualization**: Shows pitch variance, spectral centroid, and zero-crossing rate
- **Confidence Scores**: Displays prediction confidence with visual progress bar
- **Modern UI**: Responsive, gradient-based design that works on mobile and desktop

## 📋 Prerequisites

Before running this project, make sure you have:

- **Python 3.7+** installed ([Download here](https://www.python.org/downloads/))
- **pip** (Python package manager, comes with Python)
- A modern web browser (Chrome, Firefox, Safari, or Edge)
- A microphone (and optional webcam)

## 🚀 Quick Start

### 1. Download the Project

```bash
# If using Git
git clone <your-repo-url>
cd singing-vs-talking-classifier

# Or simply download and extract the ZIP file
```

### 2. Install Python Dependencies

Open your terminal/command prompt in the project folder and run:

```bash
pip install flask flask-cors librosa numpy scikit-learn
```

**Note for Windows users**: If you encounter issues installing `librosa`, you may need to install additional audio libraries:

```bash
pip install soundfile
```

### 3. Start the Backend Server

```bash
python backend.py
```

You should see:
```
* Running on http://127.0.0.1:727
```

**Keep this terminal window open!**

### 4. Open the Frontend

Simply double-click `index.html` or open it in your browser:

```bash
# On Mac/Linux
open index.html

# On Windows
start index.html

# Or manually: Right-click index.html → Open with → Your Browser
```

### 5. Use the App

1. **Toggle camera** if you want video (optional)
2. Click **"Start Recording"** and allow microphone/camera permissions
3. **Talk or sing** for a few seconds (at least 2-3 seconds)
4. Click **"Stop Recording"**
5. Wait for analysis and view your results!

## 📁 Project Structure

```
singing-vs-talking-classifier/
├── index.html          # Frontend web interface
├── backend.py          # Flask API server
├── README.md          # This file
└── requirements.txt   # Python dependencies (optional)
```

## 🔧 Troubleshooting

### "Cannot access microphone/camera"
- Make sure you **allowed permissions** when prompted
- Check browser settings: `chrome://settings/content/microphone`
- Try a different browser

### "Classification failed"
- Ensure the backend is running (check terminal)
- Verify the backend URL in browser console
- Check if port 727 is available

### Backend won't start
```bash
# Port already in use? Change port in backend.py:
app.run(debug=True, port=8000)  # Use any available port

# Then update frontend URL accordingly
```

### `librosa` installation fails
```bash
# Try installing with conda instead:
conda install -c conda-forge librosa

# Or use a simpler audio library for testing
```

### Mock mode only (not using ML)
The current version uses mock predictions for demonstration. To use real ML:
1. Collect training data (labeled singing/talking audio samples)
2. Train a model using scikit-learn
3. Save with `pickle.dump(model, open('model.pkl', 'wb'))`
4. Uncomment model loading lines in `backend.py`

## 🎓 How It Works

### Audio Features Extracted:
- **MFCCs**: Mel-frequency cepstral coefficients (voice characteristics)
- **Pitch**: Mean and variance of fundamental frequency
- **Spectral Centroid**: "Brightness" of sound
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Zero Crossing Rate**: How often signal changes sign (noisiness)

### Classification:
- Features are extracted using `librosa`
- Random Forest classifier predicts singing vs talking
- Confidence score shows prediction certainty

## 🔒 Privacy

- All processing happens locally on your machine
- No audio data is sent to external servers
- No data is stored or saved

## 🛠️ Customization

### Change Backend Port
Edit `backend.py`:
```python
app.run(debug=True, port=YOUR_PORT)
```

### Adjust Recording Quality
Edit `index.html` constraints:
```javascript
video: { width: 1280, height: 720 }  // Higher quality
```

### Modify Features
Edit `extract_features()` in `backend.py` to add/remove audio features

## 📦 Optional: Create requirements.txt

Create a file named `requirements.txt`:
```
flask==3.0.0
flask-cors==4.0.0
librosa==0.10.1
numpy==1.24.3
scikit-learn==1.3.0
soundfile==0.12.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

## ⚠️ Known Limitations

- **Mock Predictions**: Current version uses random classification (train a real model for accurate results)
- **Browser Support**: Requires modern browser with MediaRecorder API
- **Audio Format**: Works best with clear, isolated vocals
- **Short Clips**: Needs at least 2-3 seconds of audio

## 🚀 Next Steps

To make this production-ready:

1. **Train a Real Model**:
   - Collect labeled dataset of singing/talking samples
   - Train Random Forest or Deep Learning model
   - Achieve >90% accuracy on test set

2. **Improve Features**:
   - Add chroma features for musical pitch
   - Include tempo and rhythm analysis
   - Use deep learning embeddings

3. **Deploy Online**:
   - Host backend on Heroku/Railway/Render
   - Deploy frontend on GitHub Pages/Netlify
   - Add HTTPS for microphone access

4. **Add Features**:
   - Save recordings
   - Multiple language support
   - Real-time visualization of audio waveform

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

## 💡 Questions?

If something doesn't work, check:
1. Is Python installed? (`python --version`)
2. Are dependencies installed? (`pip list`)
3. Is backend running? (check terminal)
4. Did you allow microphone permissions?

---

**Happy Classifying!** 🎵💬
