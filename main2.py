import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Define constants
VIDEO_WIDTH = 420
VIDEO_HEIGHT = 300

# Directories and Parameters
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Labels dictionary for prediction
labels_dict = {0: 'Bhupesh', 1: 'Madhav', 2: 'Sai'}

# Camera control variables
# Camera control variables
cap = None
camera_on = False
predicted_character = ""
sentence_making = ""
last_sentence = ""
detection_start_time = 0
detection_threshold = 0.8
hand_detected = False
last_added_word = ""



# Global variables for prediction
current_word_idx = 0
words_list = []
video_paths = []
video_thread = None
is_paused = False

def start_camera():
    global cap, camera_on
    if not camera_on:
        camera_indices = [0, 1, 2]
        for index in camera_indices:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"Camera opened successfully with index {index}")
                camera_on = True
                process_camera_feed()
                break
        else:
            print("Error: Could not open video stream or file")
            messagebox.showerror("Error", "Unable to open camera")
    else:
        messagebox.showinfo("Info", "Camera is already on")

def stop_camera():
    global cap, camera_on
    if camera_on:
        camera_on = False
        cap.release()
        camera_feed.config(image='')

def process_camera_feed():
    global sentence_making, last_sentence, hand_detected, last_added_word
    global predicted_character, detection_start_time

    if not camera_on:
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        return

    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            expected_features = 84
            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            prediction = model.predict([np.asarray(data_aux)])
            current_predicted_character = prediction[0]

            if current_predicted_character != predicted_character:
                predicted_character = current_predicted_character
                detection_start_time = time.time()
                predicted_label.config(text=f"Predicted: {predicted_character}")

            if predicted_character != "":
                if time.time() - detection_start_time >= detection_threshold:
                    if predicted_character != last_added_word:
                        sentence_making += f" {predicted_character}"
                        last_added_word = predicted_character

    else:
        hand_detected = False
        if time.time() - detection_start_time >= detection_threshold:
            if sentence_making.strip():
                last_sentence = sentence_making.strip()
                sentence_making = ""
                last_added_word = ""

    last_sentence_label.config(text=f"Last Sentence: {last_sentence}")
    sentence_making_label.config(text=f"Current Sentence: {sentence_making.strip()}")

    # Update the camera feed label
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=frame_pil)
    camera_feed.imgtk = imgtk
    camera_feed.configure(image=imgtk)

    if camera_on:
        camera_feed.after(10, process_camera_feed)

def process_text(text):
    text = text.lower()
    filtered_text = []

    words = text.split()
    for word in words:
        filtered_text.append(word)

    processed_words = []
    for w in filtered_text:
        path = os.path.join('static', 'words', f'{w}.mp4')
        if not os.path.exists(path):
            processed_words.extend(list(w))
        else:
            processed_words.append(w)

    return processed_words

def play_video(idx):
    global is_paused, current_word_idx, video_paths

    if idx >= len(video_paths) or is_paused:
        return

    video_path = video_paths[idx]
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and not is_paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        update_image(imgtk)
        
        time.sleep(0.03)

    cap.release()
    if not is_paused:
        play_video(idx + 1)

def update_image(imgtk):
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

def start_video_thread(start_idx=0):
    global video_thread
    video_thread = threading.Thread(target=play_video, args=(start_idx,))
    video_thread.start()

def toggle_play_pause():
    global is_paused
    is_paused = not is_paused
    play_pause_button.config(text="Play" if is_paused else "Pause")

def play_selected_word():
    global current_word_idx, is_paused
    selected_idx = listbox.curselection()
    if selected_idx:
        is_paused = False
        current_word_idx = selected_idx[0]
        start_video_thread(current_word_idx)

def generate_animation():
    global current_word_idx, words_list, video_paths
    text = text_input.get()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    words_list = process_text(text)
    video_paths = [os.path.join('static', 'words', f'{word}.mp4') for word in words_list]
    listbox.delete(0, tk.END)
    for word in words_list:
        listbox.insert(tk.END, word)
    current_word_idx = 0
    if video_paths:  # Check if there are any video paths to play
        start_video_thread(current_word_idx)

def reset_video():
    global current_word_idx, video_paths, words_list, is_paused

    if video_thread and video_thread.is_alive():
        is_paused = True
        video_thread.join()

    listbox.delete(0, tk.END)

    current_word_idx = 0
    video_paths = []
    words_list = []

    last_sentence_label.config(text='Last Sentence: ')
    sentence_making_label.config(text='Current Sentence: ')

    stop_camera()

import tkinter as tk
from tkinter import ttk

# Create the main application window
root = tk.Tk()
root.title("Sign Language Conversion")
root.geometry("1200x700")
root.configure(bg="#e6ecff")



style = ttk.Style()
style.configure("TButton", font=("Arial", 14, "bold"), padding=5)

# Heading
title_label = ttk.Label(root, text="Sign Language Conversion", font=("Arial", 28, "bold"), background="#e6ecff")
title_label.pack(pady=10)

# Main container frame
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Configure grid weights for fixed 50-50 distribution
main_frame.grid_columnconfigure(0, weight=1, uniform="equal")  # Left 50%
main_frame.grid_columnconfigure(1, weight=1, uniform="equal")  # Right 50%
main_frame.grid_rowconfigure(0, weight=1)

# Left Frame for Sign to Text
left_frame = ttk.LabelFrame(main_frame, text="Sign Language to Text", padding=(10, 10), style="LeftFrame.TLabelframe")
left_frame.grid(row=0, column=0, padx=14, pady=10, sticky="nsew")

# Right Frame for Text to Animation
right_frame = ttk.LabelFrame(main_frame, text="Text to Animation", padding=(10, 10), style="RightFrame.TLabelframe")
right_frame.grid(row=0, column=1, padx=14, pady=10, sticky="nsew")

# Sign to Text Section (Left)
camera_feed_frame = ttk.Frame(left_frame, width=400, height=300)  # Set specific width and height
camera_feed_frame.pack(fill=tk.BOTH, expand=True)

camera_feed = ttk.Label(camera_feed_frame)
camera_feed.pack(fill=tk.BOTH, expand=True)

predicted_label = ttk.Label(left_frame, text="Predicted: ", font=("Arial", 14), background="#e1f5fe")
predicted_label.pack(pady=2, fill=tk.X, anchor='w')

sentence_making_label = ttk.Label(left_frame, text="Current Sentence: ", font=("Arial", 14), background="#e1f5fe")
sentence_making_label.pack(pady=2, fill=tk.X, anchor='w')

last_sentence_label = ttk.Label(left_frame, text="Last Sentence: ", font=("Arial", 14), background="#e1f5fe")
last_sentence_label.pack(pady=2, fill=tk.X, anchor='w')

# Camera control buttons
button_frame = ttk.Frame(left_frame, padding=(10, 10))
button_frame.pack(fill=tk.X)

camera_on_button = ttk.Button(button_frame, text="Camera On", command=start_camera, style="TButton")
camera_on_button.pack(side=tk.LEFT, padx=5)

camera_off_button = ttk.Button(button_frame, text="Camera Off", command=stop_camera, style="TButton")
camera_off_button.pack(side=tk.LEFT, padx=5)

# Text to Animation Section (Right)
video_frame = ttk.Frame(right_frame)
video_frame.pack(fill=tk.BOTH, expand=True)

video_label = ttk.Label(video_frame)
video_label.pack(expand=True)

# Controls and Input
controls_frame = ttk.Frame(right_frame, padding=(10, 10), style="ControlsFrame.TFrame")
controls_frame.pack(pady=10, fill=tk.X)

text_input = ttk.Entry(controls_frame, font=("Arial", 14), width=50)
text_input.pack(side=tk.TOP, padx=7, pady=2, fill=tk.X)

buttons_frame = ttk.Frame(controls_frame)
buttons_frame.pack(side=tk.TOP, pady=2)

# Add buttons with black text and hover effect
play_pause_button = ttk.Button(buttons_frame, text="Play", command=toggle_play_pause, style="TButton")
play_pause_button.pack(side=tk.LEFT, padx=10)

selected_word_button = ttk.Button(buttons_frame, text="Play Selected Word", command=play_selected_word, style="TButton")
selected_word_button.pack(side=tk.LEFT, padx=10)

generate_button = ttk.Button(buttons_frame, text="Generate", command=generate_animation, style="TButton")
generate_button.pack(side=tk.LEFT, padx=10)



# Listbox for words
listbox_frame = ttk.LabelFrame(right_frame, text="Word List", padding=(2, 2), style="ListboxFrame.TLabelframe")
listbox_frame.pack(padx=2, pady=2)

listbox = tk.Listbox(listbox_frame, font=("Arial", 14), bd=2, relief="solid", width=30, height=3)
listbox.pack(padx=2, pady=2)

# Updated styles
style = ttk.Style()
style.configure("LeftFrame.TLabelframe",
                background="#bbdefb",  # Light blue background
                font=("Arial", 14, "bold"))
style.configure("RightFrame.TLabelframe",
                background="#c8e6c9",  # Light green background
                font=("Arial", 14, "bold"))
style.configure("ControlsFrame.TFrame",
                background="#f1f8e9")  # Light yellow background
style.configure("ListboxFrame.TLabelframe",
                background="#ffe0b2",  # Light orange background
                font=("Arial", 14, "bold"))

# Start the GUI loop
root.mainloop()
