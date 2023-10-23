import numpy as np
import librosa
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

def extract_mfcc(file_name):
    y,sr = librosa.load(file_name,duration=3,offset=0.5)
    mfcc =np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    mfcc=np.expand_dims(mfcc,axis=0)
    return mfcc


st.title("Speech Emotion Recognition Web App")

audio=st.file_uploader("Choose the Audio File ['wav','mp3']")

if st.button('Speech Emotion'):
    if audio is not None:
        feature = extract_mfcc(audio)

        # Reshaping the feature to match the input shape defined while initializing the model
        feature = feature.reshape(1,40)  # Reshape to (1, num_features)

        # Prediction
        speech_emotion = model.predict(feature)

        # Probability Distribution
        emotion_index = np.argmax(speech_emotion)
        
        # Emotion Labels
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps']

        # Getting the label of the predicted emotion
        predicted_emotion_label = emotion_labels[emotion_index]

        st.title(f"{predicted_emotion_label}")