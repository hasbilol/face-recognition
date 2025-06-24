import cv2
import os
import json
import numpy as np
from datetime import datetime
import threading
from tkinter import *
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk

import torch
from torchvision import transforms
import torch.nn.functional as F

# ---------------------------
# Emotion Model Definitions
# ---------------------------
class PreprocessInput(torch.nn.Module):
    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.flip(x, dims=(0,))
        x[0, :, :] -= 91.4953
        x[1, :, :] -= 103.8827
        x[2, :, :] -= 131.0912
        return x

class Bottleneck(torch.nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = torch.nn.ReLU()
        self.downsample = i_downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=7):
        super().__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)

class LSTMPyTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = torch.nn.LSTM(512, 256, batch_first=True)
        self.fc = torch.nn.Linear(256, 7)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return self.softmax(x)

# Load models
emotion_model = ResNet(Bottleneck, [3, 4, 6, 3])
emotion_model.load_state_dict(torch.load("FER_static_ResNet50_AffectNet.pt", map_location="cpu"))
emotion_model.eval()
lstm_model = LSTMPyTorch()
lstm_model.load_state_dict(torch.load("FER_dinamic_LSTM_Aff-Wild2.pt", map_location="cpu"))
lstm_model.eval()

DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
lstm_features = []

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    PreprocessInput()
])

def detect_emotion_from_pil(pil_image):
    global lstm_features
    img = transform_pipeline(pil_image).unsqueeze(0)
    features = F.relu(emotion_model.extract_features(img)).detach().numpy()
    if len(lstm_features) == 0:
        lstm_features = [features] * 10
    else:
        lstm_features = lstm_features[1:] + [features]
    lstm_input = torch.from_numpy(np.vstack(lstm_features)).unsqueeze(0)
    output = lstm_model(lstm_input).detach().numpy()
    label_idx = np.argmax(output)
    return DICT_EMO[label_idx], float(output[0][label_idx])

# ----------------------
# Face Recognition GUI
# ----------------------
from insightface.app import FaceAnalysis

KNOWN_FACE_DIR = "known_faces"
LOG_FILE = "detections.json"
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1)

def load_known_faces():
    db = {}
    for file in os.listdir(KNOWN_FACE_DIR):
        if file.lower().endswith(('.jpg', '.png')):
            name = file.split("_")[0]
            img = cv2.imread(os.path.join(KNOWN_FACE_DIR, file))
            faces = face_app.get(img)
            if faces:
                db.setdefault(name, []).append(faces[0].normed_embedding)
    return db

def recognize_face(embedding, db, threshold=0.75):
    best_match, best_score = "Unknown", -1
    for name, embeddings in db.items():
        for known_emb in embeddings:
            score = np.dot(embedding, known_emb)
            if score > best_score and score > threshold:
                best_match, best_score = name, score
    return best_match, best_score

def show_registration_progress(root):
    win = Toplevel(root)
    win.title("Face Registration")
    Label(win, text="Move your face in different directions...").pack(pady=10)
    bar = ttk.Progressbar(win, orient=HORIZONTAL, length=300, mode='determinate', maximum=10)
    bar.pack(pady=10)
    return win, bar

class FaceGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition + Emotion Detection")
        self.db = load_known_faces()
        self.video_label = Label(root)
        self.video_label.pack()
        Button(root, text="Register", command=self.register).pack(side=LEFT, padx=10)
        Button(root, text="Detect", command=self.detect).pack(side=LEFT, padx=10)
        Button(root, text="Clear Log", command=self.clear_log).pack(side=LEFT, padx=10)
        Button(root, text="Clear Known Faces", command=self.clear_faces).pack(side=LEFT, padx=10)
        Button(root, text="Exit", command=self.quit_app).pack(side=LEFT, padx=10)
        self.mode = None
        self.name = None
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.capture_count = 0
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret: return
        self.frame = frame.copy()
        display_frame = frame.copy()

        if self.mode == "detect":
            faces = face_app.get(display_frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                name, score = recognize_face(face.normed_embedding, self.db)
                x1, y1, x2, y2 = bbox
                roi = display_frame[y1:y2, x1:x2]
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    emotion, emo_score = detect_emotion_from_pil(face_pil)
                except Exception as e:
                    emotion, emo_score = "Error", 0.0
                label = f"{name} ({score:.2f}) | {emotion} ({emo_score:.2f})"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                log = {
                    "PersonID": name,
                    "Emotion": emotion,
                    "Confidence": round(float(score), 2),
                    "EmotionScore": round(float(emo_score or 0.0), 2),
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(log) + "\n")

        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def register(self):
        self.name = simpledialog.askstring("Register", "Enter your name:")
        if not self.name:
            return
        self.mode = "register"
        self.capture_count = 0
        threading.Thread(target=self.capture_faces_progressively).start()

    def capture_faces_progressively(self):
        existing = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(KNOWN_FACE_DIR) if f.startswith(self.name + "_")]
        count = max(existing) + 1 if existing else 1
        progress_win, bar = show_registration_progress(self.root)
        for i in range(10):
            cv2.waitKey(500)
            filename = os.path.join(KNOWN_FACE_DIR, f"{self.name}_{count}.jpg")
            cv2.imwrite(filename, self.frame)
            count += 1
            bar['value'] = i + 1
            progress_win.update_idletasks()
        progress_win.destroy()
        self.db = load_known_faces()
        self.mode = None

    def detect(self):
        self.mode = "detect"

    def clear_log(self):
        if messagebox.askyesno("Confirm", "Clear detection log?"):
            open(LOG_FILE, "w").close()

    def clear_faces(self):
        if messagebox.askyesno("Confirm", "Delete all known face images?"):
            for f in os.listdir(KNOWN_FACE_DIR):
                os.remove(os.path.join(KNOWN_FACE_DIR, f))
            self.db = load_known_faces()

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = Tk()
    app = FaceGUIApp(root)
    root.mainloop()
