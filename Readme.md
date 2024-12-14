# Vartalapp: An Inclusive Communication Platform

## 🌈 Project Overview

This innovative Sign Language Translation project is a multi-modal communication tool that bridges language barriers by seamlessly translating between sign language, voice, and text. Designed with accessibility and inclusivity in mind, the application leverages cutting-edge machine learning and computer vision technologies to facilitate real-time communication.

## ✨ Key Features

### 🔄 Flexible Input/Output Modes
- **Input Modes**:
  - Sign Language (via webcam)
  - Voice Recognition
  - Text Input

### 🎯 Output Modes
- Text-to-Speech
- Text Display

### 🚀 Unique Capabilities
- Real-time and Pre-recorded translation
- Machine Learning-powered sign recognition

## 🛠 Technical Architecture

### Libraries & Technologies
- **Computer Vision**: OpenCV (cv2)
- **Machine Learning**: 
  - scikit-learn (KNeighborsClassifier)
  - MediaPipe for hand landmark detection
- **Speech Processing**:
  - SpeechRecognition
  - gTTS (Google Text-to-Speech)
- **Data Handling**: 
  - Pandas
  - NumPy

### 🧠 Machine Learning Model
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Dataset**: Sign MNIST
- **Training Process**:
  1. Load and normalize training/testing data
  2. Train KNN classifier
  3. Evaluate model accuracy

## 🔍 How It Works

### Sign Language to Other Modes
1. Capture hand landmarks using MediaPipe
2. Preprocess landmark data
3. Predict sign using trained KNN model
4. Convert prediction to corresponding character/output

### Voice/Text to Other Modes
1. Recognize speech or accept text input
2. Process input through selected output mode
3. Display or vocalize result

## 🚦 Getting Started

### Prerequisites
- Python 3.7+
- Required libraries:
  ```bash
  pip install opencv-python mediapipe numpy scikit-learn pandas speechrecognition gtts pyaudio
  ```

### Installation
1. Clone the repository
2. Install dependencies
3. Ensure dataset files are present
4. Run `python sign_language_translator.py`

## 📦 Required Resources
- Sign MNIST dataset (`sign_mnist_train.csv`, `sign_mnist_test.csv`)
- ASL image folder (`asl_images/`)

## 🌟 Potential Improvements
### Since this is a limited version of the original model utilizing the OpenSource Datasets, there are features not commited in this MVP.
- Expand sign language vocabulary
- Use Indian Sign Language (ISL)
- Implement more advanced ML models
- Add support for full sentences
- Enhance real-time performance
- Create graphical user interface (GUI)


## 🙏 Team:
- Alisha Gupta
- Aman Srivastava
- Ishant Singh
- Raghav Gupta
- Soumyadiya Ganguly
---

**Made with ❤️ to break communication barriers :)**