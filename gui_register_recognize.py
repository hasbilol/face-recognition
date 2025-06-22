import cv2
import os
import json
import numpy as np
from datetime import datetime
import threading
from tkinter import *
from tkinter import simpledialog, messagebox, ttk   
from tkinter import Toplevel, Label, ttk
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from fer import FER

# Config
KNOWN_FACE_DIR = "known_faces"
LOG_FILE = "detections.json"
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

# Initialize models
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1)
emotion_detector = FER(mtcnn=True)

# Load known faces
def load_known_faces():
    db = {}
    for file in os.listdir(KNOWN_FACE_DIR):
        if file.lower().endswith(('.jpg', '.png')):
            name = file.split("_")[0]
            img = cv2.imread(os.path.join(KNOWN_FACE_DIR, file))
            faces = face_app.get(img)
            if faces:
                if name not in db:
                    db[name] = []
                db[name].append(faces[0].normed_embedding)
    return db

def recognize_face(embedding, db, threshold=0.75):
    best_match, best_score = "Unknown", -1
    for name, embeddings in db.items():
        for known_emb in embeddings:
            score = np.dot(embedding, known_emb)
            if score > best_score and score > threshold:
                best_match = name
                best_score = score
    return best_match, best_score

from tkinter import Toplevel, Label, ttk

def show_registration_progress(root):
    win = Toplevel(root)
    win.title("Face Registration")
    Label(win, text="Move your face slowly in different directions...\nCapturing frames...").pack(pady=10)
    bar = ttk.Progressbar(win, orient=HORIZONTAL, length=300, mode='determinate', maximum=15)
    bar.pack(pady=10)
    return win, bar


# App Class
class FaceGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition + Emotion Detection")
        self.db = load_known_faces()

        self.video_label = Label(root)
        self.video_label.pack()

        self.instruction_label = Label(root, text="")
        self.instruction_label.pack()


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
        if not ret:
            return
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
                    emotion_result = emotion_detector.top_emotion(roi)
                except Exception:
                    emotion_result = None
                emotion, emo_score = emotion_result if emotion_result else ("Neutral", 0.0)

                label = f"{name} ({score:.2f}) | {emotion}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

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
        print(f"[INFO] Registering {self.name}")
        existing = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(KNOWN_FACE_DIR)
                    if f.startswith(self.name + "_")]
        count = max(existing) + 1 if existing else 1

        # Show progress window
        progress_win, bar = show_registration_progress(self.root)
        num_photos = 15

        for i in range(num_photos):
            for j in range(3, 0, -1):
                print(f"[INFO] Capturing in {j}...")
                cv2.waitKey(500)  # short wait before each capture
            filename = os.path.join(KNOWN_FACE_DIR, f"{self.name}_{count}.jpg")
            cv2.imwrite(filename, self.frame)
            print(f"[INFO] Saved {filename}")
            count += 1
            bar['value'] = i + 1
            progress_win.update_idletasks()
            cv2.waitKey(500)

        progress_win.destroy()
        print(f"[INFO] Registration complete for {self.name}")
        self.db = load_known_faces()
        self.mode = None


    def detect(self):
        print("[INFO] Detection mode activated")
        self.mode = "detect"

    def clear_log(self):
        if messagebox.askyesno("Confirm", "Clear detection log?"):
            open(LOG_FILE, "w").close()
            print("[INFO] Log cleared")

    def clear_faces(self):
        if messagebox.askyesno("Confirm", "Delete all known face images?"):
            for f in os.listdir(KNOWN_FACE_DIR):
                os.remove(os.path.join(KNOWN_FACE_DIR, f))
            print("[INFO] Known faces cleared")
            self.db = load_known_faces()

    def quit_app(self):
        self.cap.release()
        self.root.destroy()
        print("[INFO] Quitting application")

if __name__ == '__main__':
    root = Tk()
    app = FaceGUIApp(root)
    root.mainloop()
