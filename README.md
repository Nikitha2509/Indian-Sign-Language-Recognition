# Indian Sign Language Recognition System

A real-time Indian Sign Language (ISL) recognition system that uses computer vision and deep learning to recognize hand gestures through a webcam and convert them into text and speech.

## Features

- Real-time hand detection using MediaPipe
- Single-hand gesture recognition
- Double-hand gesture recognition
- TensorFlow/Keras trained models
- Real-time prediction with confidence score
- Text output for recognized gestures
- Text-to-speech output
- Live webcam interface
- Support for multiple ISL alphabet gestures

## Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- gTTS
- playsound

## Supported Gestures

### Single-Hand Gestures

The single-hand model supports:

`C` `I` `L` `O` `U` `V`

### Double-Hand Gestures

The double-hand model supports:

`A` `B` `D` `E` `F` `G` `K` `M` `N` `P` `Q` `R` `S` `T` `W` `X` `Z`

## How It Works

The system follows these steps:

1. The webcam captures a live video frame.
2. MediaPipe detects the hand and extracts 21 hand landmarks.
3. The landmarks are normalized before prediction.
4. The appropriate TensorFlow model is selected based on the number of detected hands.
5. The model predicts the corresponding ISL gesture.
6. The predicted gesture and confidence score are displayed.
7. The recognized gesture can be converted into speech.

## Project Structure

```text
Indian-Sign-Language-Recognition/
│
├── app/
│   ├── sign_double.py
│   └── sign_single.py
│
├── src/
│   ├── models/
│   ├── train/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── dataset_builder.py
│   ├── evaluation.py
│   └── config.py
│
├── classes_single_simple.json
├── classes_double_simple.json
│
├── model_single_final.h5
├── model_double_final.h5
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

Author
Nikitha S
