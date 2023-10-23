import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

def extract_mfcc(file_name):
    y, sr = librosa.load(file_name, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    audio = request.files['file']
    
    if audio.filename == '':
        return "No selected file"

    feature = extract_mfcc(audio)

    # Reshaping the feature to match the input shape defined while initializing the model
    feature = feature.reshape(1, 40)  # Reshape to (1, num_features)

    # Prediction
    speech_emotion = model.predict(feature)

    # Probability Distribution
    emotion_index = np.argmax(speech_emotion)

    # Emotion Labels
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps']

    # Getting the label of the predicted emotion
    predicted_emotion_label = emotion_labels[emotion_index]

    return predicted_emotion_label

if __name__ == '__main__':
    app.run(debug=True)
