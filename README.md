# Face Recognition + Emotion Detection App

This is a Python-based desktop GUI application that performs **real-time face recognition** and **emotion detection** using a webcam. The app allows you to **register new faces**, **recognize them later**, and **log their detected emotions** in a JSON file.

## 🚀 Features

- Real-time **Face Recognition** using InsightFace
- Real-time **Emotion Detection** using FER library
- Saves detection results in `detections.json`
- Register faces with multiple expressions (like Face ID)
- Smoothed emotion prediction using a sliding buffer
-  GUI built using Tkinter with live video feed

---

## Architecture Overview

| Task                | Library / Model         | Description |
|---------------------|--------------------------|-------------|
| Face Detection      | `insightface` (buffalo_l) | Detect faces in webcam feed |
| Face Recognition    | `insightface`            | Compare face embeddings for ID |
| Emotion Detection   | `fer` (CNN + MTCNN)      | Predict emotions from cropped faces |
| GUI Interface       | `Tkinter` + `OpenCV`     | Desktop app with buttons and video feed |
| Logging             | `json`                   | Records Person ID, Emotion, Time |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-emotion-app.git
cd face-emotion-app
```

### 2. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Launching the App
Run:
```bash
gui_register_recognize.py
```

or 

```bash
gui_register_recognize_pretained.py
```
*pretained version model from HuggingFace 

### 2. Interface Overview

The GUI provides:

Live Video Feed with detected face boxes

Buttons:

Register Face – Add a new person to your known faces database

Detect & Recognize – Start real-time recognition and emotion detection

Clear Log – Remove saved detection logs

Clear Known Faces – Erase all registered faces

### 3. Registering a New Face

1. Click Register Face.

2. Enter the Person ID / Name when prompted.

3. Look at the camera and move your face slightly (like Face ID).

4. The app will automatically capture 15 frames.

5. Once registration completes, you’ll see a confirmation message.

Tip: Register with different expressions for better matching.

### 4. Recognizing Faces & Detecting Emotions

1. Click Detect & Recognize.

2. The app will:

- Detect faces in the frame

- Compare them to the registered faces

- Predict emotions

3. Overlays:

-Name / Unknown

-Top Emotion (smoothed over recent frames)

-Confidence

4. All detections are logged in detections.json in this format:
   {
  "person_id": "John",
  "emotion": "happy",
  "timestamp": "2025-07-01 23:30:00"
}

### 5. Clearing Data
Clear Log – Deletes detections.json.
Clear Known Faces – Removes all saved embeddings (you’ll have to re-register).

### 6. Stopping the App
Close the Tkinter window, or press Ctrl+C in the terminal.




