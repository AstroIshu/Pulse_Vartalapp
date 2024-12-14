import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pyaudio

train_file = "sign_mnist_train.csv"
test_file = "sign_mnist_test.csv"
if os.path.exists(train_file) and os.path.exists(test_file):
    print("Dataset files found!")
else:
    print("Dataset files missing!")

if os.path.exists("asl_images") and os.listdir("asl_images"):
    print("ASL images folder is accessible!")
else:
    print("ASL images folder missing or empty!")

def input_output_modes():
    print("""\033[1mThis is the limited version of the original model utilizing Free Open Source Datasets
This is to protect the privacy of our project and it's resources :) \033[0m""")
    print("Select Input Mode:")
    print("1. Sign Language")
    print("2. Voice")
    print("3. Text")
    input_mode = int(input("Enter choice (1/2/3): "))
    print("Select Output Mode:")
    print("1. Sign Language")
    print("2. Voice")
    print("3. Text")
    output_mode = int(input("Enter choice (1/2/3): "))
    print("Select Mode:")
    print("1. Real-time")
    print("2. Pre-recorded")
    mode = int(input("Enter choice (1/2): "))
    return input_mode, output_mode, mode

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")

def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice input...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition.")
            return ""

def display_asl_sign(letter):
    try:
        img_path = f"asl_images/{letter.lower()}.png"
        asl_image = cv2.imread(img_path)
        if asl_image is None:
            print(f"ASL sign for '{letter}' not found!")
            return
        cv2.imshow(f"ASL Sign for '{letter}'", asl_image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying ASL sign: {e}")

def extract_landmarks(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()
    return None

def train_model():
    print("Loading dataset...")
    train_data = pd.read_csv("sign_mnist_train.csv")
    test_data = pd.read_csv("sign_mnist_test.csv")
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return knn

def sign_to_text_or_speech(input_mode, output_mode, mode, model, hands):
    if mode == 1:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extract_landmarks(frame, hands)
            if landmarks is not None:
                recognized_text = chr(model.predict([landmarks])[0] + 65)
                if output_mode == 1:
                    display_asl_sign(recognized_text)
                elif output_mode == 2:
                    text_to_speech(recognized_text)
                elif output_mode == 3:
                    print(f"Output Text: {recognized_text}")
            cv2.imshow('Real-time Sign Language Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        video_file = input("Enter the path to the video file: ")
        if not os.path.exists(video_file):
            print("Video file not found!")
            return
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extract_landmarks(frame, hands)
            if landmarks is not None:
                recognized_text = chr(model.predict([landmarks])[0] + 65)

def main():
    input_mode, output_mode, mode = input_output_modes()
    if input_mode == 1:
        hands = mp.solutions.hands.Hands()
        model = train_model()
        sign_to_text_or_speech(input_mode, output_mode, mode, model, hands)
    elif input_mode == 2:
        recognized_text = voice_to_text()
        if output_mode == 1:
            display_asl_sign(recognized_text)
        elif output_mode == 2:
            text_to_speech(recognized_text)
        elif output_mode == 3:
            print(f"Output Text: {recognized_text}")
    elif input_mode == 3:
        text_input = input("Enter the text: ")
        if output_mode == 1:
            for char in text_input:
                display_asl_sign(char)
        elif output_mode == 2:
            text_to_speech(text_input)
        elif output_mode == 3:
            print(f"Output Text: {text_input}")

if __name__ == "__main__":
    main()
