import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import sounddevice as sd

# -----------------------------
# Helper functions
# -----------------------------

def extract_features(file_path_or_bytes):
    """Extract MFCC features from audio file (path or BytesIO)."""
    try:
        if isinstance(file_path_or_bytes, str):
            audio, sample_rate = librosa.load(file_path_or_bytes, sr=22050)
        else:
            audio, sample_rate = sf.read(file_path_or_bytes)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def load_data(data_dir):
    features, labels = [], []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            class_label = 1 if 'scream' in file.lower() else 0
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(class_label)
    return np.array(features), np.array(labels)

def train_model():
    data_dir = 'assets/positive'  # adjust if needed
    X, y = load_data(data_dir)
    if len(X) == 0:
        st.error("No data found. Please check your dataset folder path.")
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

def predict_scream(model, file_bytes):
    features = extract_features(file_bytes)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        return prediction, prob
    else:
        return None, None

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Scream Detection App", page_icon="🔊", layout="wide")
st.title("🔊 Scream Detection and Analysis System")
st.write("Upload an audio file or record live to detect scream sounds.")

# Sidebar training
st.sidebar.header("🧠 Model Training")
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        model, accuracy, report = train_model()
        if model:
            joblib.dump(model, "model.pkl")
            st.sidebar.success(f"✅ Model trained with {accuracy*100:.2f}% accuracy.")
            st.sidebar.text("Classification Report:")
            st.sidebar.text(report)
else:
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        st.sidebar.success("✅ Pretrained model loaded.")
    else:
        st.sidebar.warning("Please train the model first.")

# -----------------------------
# Upload audio for prediction
# -----------------------------
st.subheader("🎵 Upload Audio File")
audio_file = st.file_uploader("Choose an audio file (WAV or MP3)", type=["wav", "mp3"])

if audio_file and 'model' in locals():
    st.audio(audio_file)
    with st.spinner("Analyzing..."):
        prediction, probability = predict_scream(model, io.BytesIO(audio_file.read()))
        if prediction is not None:
            if prediction == 1:
                st.error(f"🚨 Scream detected! Probability: {probability:.2f}")
            else:
                st.success(f"✅ No scream detected. Probability: {probability:.2f}")
        else:
            st.warning("Could not process the file.")

# -----------------------------
# Live 2-second recording
# -----------------------------
st.subheader("🎤 Record Live Audio (2 seconds)")

if st.button("Record"):
    st.info("Recording for 2 seconds...")
    duration = 2  # seconds
    fs = 22050    # sampling rate

    try:
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        audio_data = np.squeeze(audio_data)
        
        # Convert to WAV bytes
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, audio_data, fs, format='WAV')
        wav_bytes.seek(0)
        
        st.audio(wav_bytes)

        if 'model' in locals():
            with st.spinner("Analyzing recorded audio..."):
                prediction, probability = predict_scream(model, wav_bytes)
                if prediction is not None:
                    if prediction == 1:
                        st.error(f"🚨 Scream detected! Probability: {probability:.2f}")
                    else:
                        st.success(f"✅ No scream detected. Probability: {probability:.2f}")
                else:
                    st.warning("Could not process the recorded audio.")
        else:
            st.warning("⚠️ Please train or load the model first.")
    except Exception as e:
        st.error(f"Recording failed: {e}")

# Footer
st.markdown("---")
st.caption("Developed by **Sajal Kumar Jha** | Scream Detection for Crime Control Project")
