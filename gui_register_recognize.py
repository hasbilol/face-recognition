import cv2
import time
import os
import json
import numpy as np
from datetime import datetime
import threading
from tkinter import *
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from fer import FER
from collections import deque, Counter
#F
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

# Progress bar popup with instruction
def show_registration_progress(root):
    win = Toplevel(root)
    win.title("Face Registration")

    instruction = Label(win, text="Initializing...", font=("Helvetica", 12))
    instruction.pack(pady=(10, 0))

    Label(win, text="Move your face slowly as instructed\nCapturing frames...").pack(pady=(5, 10))

    bar = ttk.Progressbar(win, orient=HORIZONTAL, length=300, mode='determinate', maximum=10)
    bar.pack(pady=10)

    return win, bar, instruction

# App Class
class FaceGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition + Emotion Detection")
        self.db = load_known_faces()

        self.prev_time = time.time()
        self.fps = 0.0

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
        self.last_emotion = ("Neutral", 0.0)
        self.emotion_buffer = deque(maxlen=5)
        self.frame_count = 0
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame = frame.copy()
        display_frame = frame.copy()

        self.frame_count += 1

        if self.mode == "detect":
            faces = face_app.get(display_frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                name, score = recognize_face(face.normed_embedding, self.db)
                x1, y1, x2, y2 = bbox

                # Add margin to ROI
                margin = 20
                x1m = max(x1 - margin, 0)
                y1m = max(y1 - margin, 0)
                x2m = min(x2 + margin, display_frame.shape[1])
                y2m = min(y2 + margin, display_frame.shape[0])
                roi = display_frame[y1m:y2m, x1m:x2m]

                # Detect emotion every 5 frames
                if self.frame_count % 5 == 0:
                    try:
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_resized = cv2.resize(roi_rgb, (48, 48))
                        emotion_result = emotion_detector.top_emotion(roi_resized)
                        if emotion_result:
                            self.emotion_buffer.append(emotion_result[0])
                            common_emotion = Counter(self.emotion_buffer).most_common(1)[0]
                            self.last_emotion = (common_emotion[0], emotion_result[1])
                    except Exception:
                        pass

                # Use smoothed emotion
                emotion, emo_score = self.last_emotion

                emo_score_display = float(emo_score) if emo_score is not None else 0.0
                label = f"{name} ({score:.2f}) | {emotion} ({emo_score_display:.2f})"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                # Log detection
                log = {
                    "PersonID": name,
                    "Emotion": emotion,
                    "Confidence": round(float(score), 2),
                    "EmotionScore": round(float(emo_score or 0.0), 2),
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(log) + "\n")

        # ✅ Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        # ✅ Draw FPS on frame
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(display_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

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
        threading.Thread(target=self.capture_faces_progressively).start()

    def capture_faces_progressively(self):
        print(f"[INFO] Registering {self.name}")
        existing = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(KNOWN_FACE_DIR)
                    if f.startswith(self.name + "_")]
        count = max(existing) + 1 if existing else 1

        progress_win, bar, instruction_label = show_registration_progress(self.root)
        expressions = [
            "Neutral expression",
            "Smile a little 😊",
            "Smile widely 😄",
            "Look to your left 👈",
            "Look to your right 👉",
            "Look up ⬆️",
            "Look down ⬇️",
            "Raise your eyebrows 😯",
            "Close your eyes gently 😌",
            "Relax your face again"
        ]

        for i in range(len(expressions)):
            instruction_label.config(text=f"Step {i+1}: {expressions[i]}")
            instruction_label.update()

            for j in range(3, 0, -1):
                print(f"[INFO] Capturing in {j}...")
                cv2.waitKey(500)
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
