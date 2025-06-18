import cv2
import os
import json
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

# === Load Models ===

# Face embedding model (ArcFace)
embed_processor = AutoImageProcessor.from_pretrained("nateraw/arcface-resnet100-asia")
embed_model = AutoModel.from_pretrained("nateraw/arcface-resnet100-asia")
embed_model.eval()

# Emotion detection model (FER+)
emotion_processor = AutoImageProcessor.from_pretrained("nateraw/ferplus")
emotion_model = AutoModelForImageClassification.from_pretrained("nateraw/ferplus")
emotion_model.eval()

# === Face recognition helpers ===

def get_embedding(pil_image):
    inputs = embed_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb

def recognize_face(embedding, known_embeddings, threshold=0.6):
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    best_match, best_score = "Unknown", 0
    for name, known_emb in known_embeddings.items():
        score = cosine_sim(embedding, known_emb)
        if score > best_score and score > threshold:
            best_match, best_score = name, score
    return best_match

def detect_emotion(pil_image):
    inputs = emotion_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return emotion_model.config.id2label[pred]

def load_known_faces():
    db = {}
    for file in os.listdir("known_faces"):
        path = os.path.join("known_faces", file)
        img = Image.open(path).convert("RGB")
        emb = get_embedding(img)
        name = os.path.splitext(file)[0]
        db[name] = emb.tolist()
    with open("embeddings.json", "w") as f:
        json.dump(db, f)

# === Load or generate embeddings.json ===
if not os.path.exists("embeddings.json"):
    load_known_faces()

with open("embeddings.json", "r") as f:
    known_embeddings = json.load(f)

# === Webcam feed ===
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = rgb[y:y+h, x:x+w]
        pil_face = Image.fromarray(face_crop).convert("RGB")

        emb = get_embedding(pil_face)
        person = recognize_face(emb, known_embeddings)
        emotion = detect_emotion(pil_face)

        label = f"{person} ({emotion})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        log = {
            "PersonID": person,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Emotion": emotion
        }
        with open("output.json", "a") as f:
            f.write(json.dumps(log) + "\n")

    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
