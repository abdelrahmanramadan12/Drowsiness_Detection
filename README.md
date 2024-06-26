# Drowsiness_Detection

This project implements a drowsiness detection system using computer vision and facial landmark detection techniques. It continuously monitors a person's eyes and triggers an alarm if signs of drowsiness are detected.

## Features

- Real-time eye aspect ratio (EAR) calculation.
- Detection of drowsiness based on EAR threshold.
- Alerts using audio signals.
- Serial communication to send alerts.

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- scipy
- pygame

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/drowsiness-detection.git
    cd drowsiness-detection
    ```

2. Install the required Python packages:
    ```bash
    pip install opencv-python dlib imutils scipy pygame
    ```

3. Download the required dlib model for facial landmark detection and place it in the `models` directory:
    ```bash
    mkdir -p models
    wget -O models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2
    ```

## Usage

1. Connect your camera and ensure it is recognized by your system.
2. Run the drowsiness detection script:
    ```bash
    python Drowsiness_Detection.py
    ```

## How It Works

1. **Eye Aspect Ratio (EAR) Calculation**: The EAR is computed for both eyes. If the EAR falls below a certain threshold for a predefined number of consecutive frames, the person is considered to be drowsy.
2. **Facial Landmark Detection**: The system uses dlib's pre-trained facial landmark detector to locate the eyes in the video feed.
3. **Alert Mechanism**: An audio alert is triggered using the `pygame` library if drowsiness is detected.
4. **Serial Communication**: The system sends a signal over a serial port to indicate the detection of drowsiness.

## Customization

- **Threshold and Frame Check**: The drowsiness detection threshold and the number of consecutive frames can be adjusted to suit different requirements.
    ```python
    thresh = 0.25
    frame_check = 20
    ```

- **Serial Port Configuration**: Update the serial port configuration based on your setup.
    ```python
    s = serial.Serial('COM3', 9600)
    ```

