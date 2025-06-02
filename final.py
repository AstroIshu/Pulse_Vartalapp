import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
from collections import deque

# Load hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load trained ASL model
MODEL_FILE = "asl_cnn_model.h5"
IMG_SIZE = 28
model = load_model(MODEL_FILE)

# Prediction settings
prediction_delay = 2.0  # Delay in seconds between predictions
previous_predictions = deque(maxlen=10)  # Store last 10 predictions
last_prediction_time = time.time()
stable_prediction = None

# Predict ASL sign
def predict_asl_sign(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return chr(predicted_label + 65)  # Convert to ASCII letter

# Function to draw a scanning spiral effect
def draw_spiral(img, center_x, center_y, radius=80, num_turns=5):
    for i in range(1, num_turns * 10):
        angle = i * (2 * np.pi / 10)
        r = (radius * i) / (num_turns * 10)
        x = int(center_x + r * np.cos(angle))
        y = int(center_y + r * np.sin(angle))
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# Function to draw a net (grid) around hand
def draw_net_effect(img, landmarks):
    if landmarks:
        x_min = min([lm.x for lm in landmarks]) * img.shape[1]
        x_max = max([lm.x for lm in landmarks]) * img.shape[1]
        y_min = min([lm.y for lm in landmarks]) * img.shape[0]
        y_max = max([lm.y for lm in landmarks]) * img.shape[0]
        
        # Draw vertical and horizontal lines
        step_x = (x_max - x_min) / 6
        step_y = (y_max - y_min) / 6

        for i in range(7):
            x = int(x_min + i * step_x)
            y = int(y_min + i * step_y)
            cv2.line(img, (x, int(y_min)), (x, int(y_max)), (255, 0, 0), 1)
            cv2.line(img, (int(x_min), y), (int(x_max), y), (255, 0, 0), 1)

# Real-time ASL recognition with stabilized predictions
def real_time_recognition():
    global last_prediction_time, stable_prediction
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    translation_window = np.zeros((200, 400, 3), dtype=np.uint8)  # Black background

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Flip frame for a mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the detected hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get palm center
                    palm_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                    palm_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                    # Draw scanning effects
                    draw_spiral(frame, palm_x, palm_y, radius=80, num_turns=5)
                    draw_net_effect(frame, hand_landmarks.landmark)

            if hand_detected:
                letter = predict_asl_sign(frame)
                previous_predictions.append(letter)

                # Calculate the most frequent prediction in recent frames
                most_common_prediction = max(set(previous_predictions), key=previous_predictions.count)

                # Apply delay-based stability filter
                if most_common_prediction != stable_prediction:
                    last_prediction_time = time.time()
                    stable_prediction = most_common_prediction
                elif time.time() - last_prediction_time > prediction_delay:
                    print(f"Predicted: {stable_prediction}")

            # Show predicted letter in a separate window
            translation_window[:] = (0, 0, 0)  # Reset to black
            cv2.putText(translation_window, f"Translation: {stable_prediction if stable_prediction else ''}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("ASL Recognition", frame)
            cv2.imshow("Translation", translation_window)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exiting ASL Recognition...")
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run real-time ASL recognition
real_time_recognition()










































































# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load hand tracking module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Load trained ASL model
# MODEL_FILE = "asl_cnn_model.h5"
# IMG_SIZE = 28
# model = load_model(MODEL_FILE)

# # Predict ASL sign
# def predict_asl_sign(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     image = img_to_array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     prediction = model.predict(image)
#     return chr(np.argmax(prediction) + 65)

# # Function to draw a scanning spiral effect
# def draw_spiral(img, center_x, center_y, radius=80, num_turns=5):
#     for i in range(1, num_turns * 10):
#         angle = i * (2 * np.pi / 10)
#         r = (radius * i) / (num_turns * 10)
#         x = int(center_x + r * np.cos(angle))
#         y = int(center_y + r * np.sin(angle))
#         cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# # Function to draw a net (grid) around hand
# def draw_net_effect(img, landmarks):
#     if landmarks:
#         x_min = min([lm.x for lm in landmarks]) * img.shape[1]
#         x_max = max([lm.x for lm in landmarks]) * img.shape[1]
#         y_min = min([lm.y for lm in landmarks]) * img.shape[0]
#         y_max = max([lm.y for lm in landmarks]) * img.shape[0]
        
#         # Draw vertical and horizontal lines
#         step_x = (x_max - x_min) / 6
#         step_y = (y_max - y_min) / 6

#         for i in range(7):
#             x = int(x_min + i * step_x)
#             y = int(y_min + i * step_y)
#             cv2.line(img, (x, int(y_min)), (x, int(y_max)), (255, 0, 0), 1)
#             cv2.line(img, (int(x_min), y), (int(x_max), y), (255, 0, 0), 1)

# # Real-time ASL recognition with effects
# def real_time_recognition():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     translation_window = np.zeros((200, 400, 3), dtype=np.uint8)  # Black background

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture image")
#                 break

#             # Flip frame for a mirror effect
#             frame = cv2.flip(frame, 1)

#             # Convert to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw the detected hand landmarks
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                     # Get palm center
#                     palm_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
#                     palm_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

#                     # Draw scanning effects
#                     draw_spiral(frame, palm_x, palm_y, radius=80, num_turns=5)
#                     draw_net_effect(frame, hand_landmarks.landmark)

#             # Predict letter (only if a hand is detected)
#             letter = predict_asl_sign(frame)
#             print(f"Predicted: {letter}")

#             # Show predicted letter in a separate window
#             translation_window[:] = (0, 0, 0)  # Reset to black
#             cv2.putText(translation_window, f"Translation: {letter}", (50, 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             cv2.imshow("ASL Recognition", frame)
#             cv2.imshow("Translation", translation_window)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 print("Exiting ASL Recognition...")
#                 break

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# # Run real-time ASL recognition
# real_time_recognition()
