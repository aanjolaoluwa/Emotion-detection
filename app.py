# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and labels
MODEL_PATH = 'face_emotionModel.h5'
LABELS_PATH = 'class_labels.json'

model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

# Database setup
DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Emotion messages
CHEER_MESSAGES = {
    'sad': "You are frowning. Why are you sad? Don’t worry — every storm passes. You’re stronger than you know!",
    'angry': "You look angry. Take a deep breath. Tomorrow is a new day!",
    'fear': "You seem scared. It’s okay — you’ve got this. One step at a time!",
    'disgust': "Hmm, something bothering you? Let it go — peace feels better!",
}

def predict_emotion(image_path):
    # Load and preprocess image
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 48, 48, 1)
    img_array = img_array.astype('float32') / 255.0

    # Predict
    prediction = model.predict(img_array)
    emotion_idx = np.argmax(prediction)
    emotion = class_labels[emotion_idx]
    confidence = float(np.max(prediction))
    return emotion, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        file = request.files['photo']

        if file and name and email:
            # Save image
            filename = f"{email.split('@')[0]}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Predict emotion
            emotion, confidence = predict_emotion(image_path)

            # Cheer message
            message = CHEER_MESSAGES.get(emotion, f"You are {emotion}. Great job!")

            # Save to DB
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, image_path, emotion) VALUES (?, ?, ?, ?)",
                      (name, email, image_path, emotion))
            conn.commit()
            conn.close()

            return render_template('index.html',
                                   message=message,
                                   emotion=emotion.capitalize(),
                                   confidence=round(confidence * 100, 1),
                                   image_url=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)