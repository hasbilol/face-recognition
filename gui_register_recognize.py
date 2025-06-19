import cv2
import os
import json
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Setup face engine
face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_engine.prepare(ctx_id=-1)

KNOWN_FACE_DIR = "known_faces"
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

# Load known faces
def load_known_faces():
    db = {}
    for file in os.listdir(KNOWN_FACE_DIR):
        if file.lower().endswith((".jpg", ".png")):
            name = file.split("_")[0]
            img = cv2.imread(os.path.join(KNOWN_FACE_DIR, file))
            faces = face_engine.get(img)
            if faces:
                if name not in db:
                    db[name] = []
                db[name].append(faces[0].normed_embedding)
    return db

# Recognize face
def recognize_face(embedding, db, threshold=0.75):
    best_match, best_score = "Unknown", -1
    for name, embeddings in db.items():
        for known_emb in embeddings:
            score = np.dot(embedding, known_emb)
            if score > best_score and score > threshold:
                best_match = name
                best_score = score
    return best_match, best_score

class FaceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")
        self.cap = cv2.VideoCapture(0)
        self.known_faces_db = load_known_faces()
        self.mode = None
        self.current_name = None
        self.sample_count = 0

        self.video_label = tk.Label(master)
        self.video_label.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Register", command=self.start_register).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Detect", command=self.start_detect).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Clear Faces", command=self.clear_faces).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Exit", command=self.exit_app).pack(side=tk.LEFT)

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame = frame.copy()

        if self.mode == "register" and self.sample_count < 3:
            faces = face_engine.get(frame)
            if faces:
                face = faces[0]
                existing = [f for f in os.listdir(KNOWN_FACE_DIR) if f.startswith(self.current_name)]
                filename = os.path.join(KNOWN_FACE_DIR, f"{self.current_name}_{len(existing)+1}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Saved {filename}")
                self.sample_count += 1
                if self.sample_count == 3:
                    messagebox.showinfo("Registration", f"Registration complete for {self.current_name}")
                    self.known_faces_db = load_known_faces()
                    self.mode = None

        elif self.mode == "detect":
            faces = face_engine.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                emb = face.normed_embedding
                name, score = recognize_face(emb, self.known_faces_db)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{name} ({score:.2f})", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                log = {
                    "PersonID": name,
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Confidence": round(float(score), 2)
                }
                with open("detections.json", "a") as f:
                    f.write(json.dumps(log) + "\n")

        # Convert image for Tkinter
        image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.video_label.configure(image=image)
        self.video_label.image = image
        self.master.after(10, self.update_video)

    def start_register(self):
        name = simpledialog.askstring("Register", "Enter your name:")
        if name:
            self.current_name = name.strip()
            self.sample_count = 0
            self.mode = "register"

    def start_detect(self):
        self.mode = "detect"

    def clear_log(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear detections log?"):
            open("detections.json", "w").close()
            messagebox.showinfo("Log Cleared", "detections.json has been cleared.")

    def clear_faces(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all known faces?"):
            for file in os.listdir(KNOWN_FACE_DIR):
                file_path = os.path.join(KNOWN_FACE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self.known_faces_db = load_known_faces()
            messagebox.showinfo("Faces Cleared", "All known face images have been deleted.")

    def exit_app(self):
        self.cap.release()
        self.master.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    face_app = FaceApp(root)
    root.mainloop()
