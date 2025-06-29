# 🧠 Face Recognition + Emotion Detection App

This is a Python-based desktop GUI application that performs **real-time face recognition** and **emotion detection** using a webcam. The app allows you to **register new faces**, **recognize them later**, and **log their detected emotions** in a JSON file.

## 🚀 Features

- ✅ Real-time **Face Recognition** using InsightFace
- 😊 Real-time **Emotion Detection** using FER library
- 💾 Saves detection results in `detections.json`
- 🧍‍♂️ Register faces with multiple expressions (like Face ID)
- 📊 Smoothed emotion prediction using a sliding buffer
- 🖥️ GUI built using Tkinter with live video feed

---

## 🏗️ Architecture Overview

| Task                | Library / Model         | Description |
|---------------------|--------------------------|-------------|
| Face Detection      | `insightface` (buffalo_l) | Detect faces in webcam feed |
| Face Recognition    | `insightface`            | Compare face embeddings for ID |
| Emotion Detection   | `fer` (CNN + MTCNN)      | Predict emotions from cropped faces |
| GUI Interface       | `Tkinter` + `OpenCV`     | Desktop app with buttons and video feed |
| Logging             | `json`                   | Records Person ID, Emotion, Time |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-emotion-app.git
cd face-emotion-app
