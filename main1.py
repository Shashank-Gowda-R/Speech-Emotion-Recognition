import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request, redirect, flash, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from cryptography.fernet import Fernet
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = '5299a1baf9ac8e572dcc5b8d1d3972b0f09f5a71bf720a0f9590bb32186cd2ff'  # Replace with a strong, random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database for user information
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to store uploaded audio files
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

model = tf.keras.models.load_model('model.h5')

# User model for the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    encrypted_audio = db.Column(db.LargeBinary)  # Store the encrypted audio in the database

# Create a Fernet object with the encryption key
key = b'abs-ZI77tSXUKLKcPTEcprMtcXCmlKk7ww1Y5HwoDT0='
fernet = Fernet(key)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def extract_mfcc(file_name):
    y, sr = librosa.load(file_name, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

@app.route('/index')
@login_required
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
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

    # Encrypt and store the audio in the database
    encrypted_audio = fernet.encrypt(audio.read())

    # Store the encrypted audio in the database
    user = User.query.get(int(current_user.get_id()))  # Get the current user
    user.encrypted_audio = encrypted_audio
    db.session.commit()

    flash('Audio file uploaded and analyzed successfully!', 'success')

    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user is not None and user.password == password:
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            user = User(username=username, password=password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Password and Confirm Password do not match.', 'error')

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.template_filter('fernet_decrypt')
def fernet_decrypt(value):
    return fernet.decrypt(value).decode()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
