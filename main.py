import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pyaudio
import wave
import smtplib

# Send email if scream is detected
def send_email(probability):
    try:
        print('\nSMTP')
        HOST = "smtp.gmail.com"
        PORT = 587
        FROM_EMAIL = "xxxxxxxx@gmail.com"
        PASSWORD = "xxxxxxxxxxxxxxxx"  # Replace with app password
        TO_EMAIL = "xxxxxxxxxxx@gmail.com"
        subject = 'Human Scream Detection Project'
        msg = f"Scream detected with probability: {probability:.2f}"

        MESSAGE = f"Subject: {subject}\n\n{msg}"

        smtp = smtplib.SMTP(HOST, PORT)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(FROM_EMAIL, PASSWORD)
        smtp.sendmail(FROM_EMAIL, TO_EMAIL, MESSAGE)
        smtp.quit()
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

# Feature extraction
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        messagebox.showerror("Error", f"Error processing file: {file_name}")
        return None

# Load dataset
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

# Record audio
def record_audio(filename, duration=2, fs=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = [stream.read(1024) for _ in range(int(fs / 1024 * duration))]
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Predict function
def predict_scream_from_file(model, file_name):
    features = extract_features(file_name)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        prob = model.predict_proba(features)
        return prediction[0], prob[0][1]
    else:
        return None, None

# Load and train model
data_dir = 'Assets/positive'
X, y = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# GUI
class ScreamDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Scream Detection")
        self.master.geometry("800x600")

        self.canvas = tk.Canvas(self.master, height=600, width=800)
        self.canvas.pack()

        self.frame = tk.Frame(self.master, bg="white")
        self.frame.place(relwidth=1, relheight=1)

        self.title_label = ttk.Label(self.frame, text="Scream Detection App", font=("Helvetica", 18))
        self.title_label.pack(pady=20)

        self.record_button = ttk.Button(self.frame, text="Record Audio", command=self.record_audio, width=20)
        self.record_button.pack(pady=10)

        self.predict_button = ttk.Button(self.frame, text="Predict Scream", command=self.predict_scream, width=20)
        self.predict_button.pack(pady=10)

        self.select_button = ttk.Button(self.frame, text="Select Audio", command=self.select_audio, width=20)
        self.select_button.pack(pady=10)

        self.test_button = ttk.Button(self.frame, text="Test Selected Audio", command=self.test_selected_audio, width=20)
        self.test_button.pack(pady=10)

        self.result_label = ttk.Label(self.frame, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        self.smtp_label = ttk.Label(self.frame, text="", font=("Helvetica", 14))
        self.smtp_label.pack(pady=20)

        self.accuracy_label = ttk.Label(self.frame, text=f"Accuracy: {accuracy:.2f}", font=("Helvetica", 14))
        self.accuracy_label.pack(pady=10)

        self.report_label = tk.Text(self.frame, height=10, width=80)
        self.report_label.pack(pady=10)
        self.report_label.insert(tk.END, report)
        self.report_label.config(state=tk.DISABLED)

        self.selected_file = None

    def record_audio(self):
        filename = 'output.wav'
        record_audio(filename)
        messagebox.showinfo("Record Audio", "Audio recorded successfully.")

    def predict_scream(self):
        filename = 'output.wav'
        prediction, probability = predict_scream_from_file(model, filename)
        if prediction is not None:
            if prediction == 1:
                self.result_label.config(text=f"Scream detected with probability: {probability:.2f}")
            else:
                self.result_label.config(text=f"No scream detected. Probability of scream: {probability:.2f}")
        else:
            messagebox.showerror("Predict Scream", "Prediction error.")

    def select_audio(self):
        self.selected_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.selected_file:
            messagebox.showinfo("Select Audio", f"Selected file: {self.selected_file}")

    def test_selected_audio(self):
        if self.selected_file:
            prediction, probability = predict_scream_from_file(model, self.selected_file)
            if prediction is not None:
                if prediction == 1:
                    self.result_label.config(text=f"Scream detected with probability: {probability:.2f}")
                    if send_email(probability):
                        self.smtp_label.config(text="Email was successfully sent!")
                    else:
                        self.smtp_label.config(text="Email sending failed.")
                else:
                    self.result_label.config(text=f"No scream detected. Probability of scream: {probability:.2f}")
            else:
                messagebox.showerror("Test Selected Audio", "Error occurred during prediction.")
        else:
            messagebox.showwarning("Test Selected Audio", "Please select an audio file.")

def main():
    root = tk.Tk()
    app = ScreamDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
