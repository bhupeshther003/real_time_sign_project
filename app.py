# import language_tool_python
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import pickle
from email.message import EmailMessage
import ssl
import smtplib
import os
import random
from PIL import Image
import cv2
import mediapipe as mp
from flask_cors import CORS
import subprocess 
from tensorflow.keras.models import load_model
import os
import secrets

# Load spaCy model
# nlp = spacy.load('en_core_web_sm')



app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Ensure this folder exists
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}

# Load your model
model = load_model('sign_language_cnn_model_word.h5')  # Replace with your model file path

# Load class labels
class_labels = np.load('classes.npy', allow_pickle=True).tolist()  # Ensure this is a list or dict
print("Class Labels List:", class_labels)  # Debug print


DEFAULT_SENDER_EMAIL = 'werqcontact@gmail.com'
DEFAULT_SENDER_PASSWORD = 'cubb gogr dfii evqs'


CORS(app)  # Enable CORS
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/communication')
# def services():
#     return render_template('services.html')

# @app.route('/career')
# def career():
#     return render_template('https://job-portal-2-p5bv.onrender.com/')
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email_receiver = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        body = f"Name: {name}\nEmail: {email_receiver}\nMessage: {message}"

        try:
            send_email(DEFAULT_SENDER_EMAIL, DEFAULT_SENDER_PASSWORD, email_receiver, subject, body)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            print(f"Error: {e}")
            flash('An error occurred while sending the message. Please try again later.', 'danger')

        return redirect(url_for('contact'))

    return render_template('contact.html')

def send_email(email_sender, email_password, email_receiver, subject, body):
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


@app.route('/run_camera', methods=['POST'])
def run_camera():
    # Run the camera.py script
    try:
        subprocess.Popen(['python', 'main.py'], shell=False)
        return redirect(url_for('animation_view'))
    except Exception as e:
        return f"An error occurred: {e}"



#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++


@app.route('/communication', methods=['GET', 'POST'])
def animation_view():
    if request.method == 'POST':
        text = request.form.get('sen', '')
        text = text.lower()

        # Process the text using spaCy
        doc = nlp(text)

        # Initialize tense counters
        tense = {
            "future": 0,
            "present": 0,
            "past": 0,
            "present_continuous": 0
        }

        filtered_text = []
        for token in doc:
            # Token and part-of-speech tagging
            pos = token.pos_
            tag = token.tag_

            # Count tenses
            if tag == "MD":
                tense["future"] += 1
            elif pos in ["VERB", "AUX"]:
                if tag in ["VBG", "VBN", "VBD"]:
                    tense["past"] += 1
                elif tag == "VBG":
                    tense["present_continuous"] += 1
                else:
                    tense["present"] += 1

            # Lemmatization
            if pos in ["VERB", "NOUN"]:
                filtered_text.append(token.lemma_)
            elif pos in ["ADJ", "ADV"]:
                filtered_text.append(token.lemma_)
            else:
                filtered_text.append(token.text)

        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            filtered_text = ["Before"] + filtered_text
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in filtered_text:
                filtered_text = ["Will"] + filtered_text
        elif probable_tense == "present":
            if tense["present_continuous"] >= 1:
                filtered_text = ["Now"] + filtered_text

        # Handle static files
        processed_words = []
        for w in filtered_text:
            path = os.path.join(app.static_folder, 'words', f'{w}.mp4')
            if not os.path.exists(path):
                processed_words.extend(list(w))
            else:
                processed_words.append(w)
        filtered_text = processed_words

        return render_template('services.html', words=filtered_text, text=text)
    else:
        return render_template('services.html')

#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++




# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(frame):
    frame_resized = cv2.resize(frame, (64, 64))  # Assuming model expects 64x64 images
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    prediction = model.predict(frame_expanded)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    print("Predicted Class Index:", predicted_class_index)

    # Handle fallback
    if 0 <= predicted_class_index < len(class_labels):
        return class_labels[predicted_class_index]
    else:
        return "Unknown"

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    predictions = []

    with mp_holistic.Holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Optional: Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # Process frame for model prediction
            predicted_class = process_image(frame)
            predictions.append(predicted_class)

            # Optional: Display frame with landmarks (for debugging)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    cap.release()
    # cv2.destroyAllWindows()  # Uncomment if using OpenCV for debugging
    if predictions:
        most_common_prediction = max(set(predictions), key=predictions.count)
    else:
        most_common_prediction = "Unknown"
    return most_common_prediction


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.lower().endswith(('mp4', 'avi', 'mov')):
            prediction = process_video(filepath)
        else:
            frame = cv2.imread(filepath)
            prediction = process_image(frame)

        # Provide the URL for the uploaded file
        file_url = url_for('send_file', filename=filename)
        return render_template('result.html', result=f' Predicted sign: {prediction}', file_url=file_url)
    else:
        return "Unsupported file format"

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

