# recognize_attendance.py
import cv2
import os
import csv
import time
from datetime import datetime

TRAINER_FILE = os.path.join("trainer", "trainer.yml")
CAMERA_INDEX = 0
MIN_SIZE = (80,80)
ATTENDANCE_FILE = "attendance.csv"

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise IOError("Cannot load cascade file")

# Load recognizer
if not os.path.exists(TRAINER_FILE):
    print("Trainer file not found, run train_model.py first.")
    exit(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)

# Helper: map id -> name from dataset folders
def build_id_name_map(dataset_dir="dataset"):
    id_name = {}
    for d in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, d)
        if os.path.isdir(full) and "-" in d:
            try:
                id_ = int(d.split("-")[0])
                name = d.split("-",1)[1]
                id_name[id_] = name
            except:
                pass
    return id_name

id_name_map = build_id_name_map()

def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # avoid duplicate for same session: we will record only once per run per name
    # To keep it simple: write if not already in file today (simple check)
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            rows = list(csv.reader(f))
            for row in rows:
                if len(row) >= 2 and row[0] == name and row[1].split(" ")[0] == now.split(" ")[0]:
                    return False
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now])
    return True

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open webcam.")
    exit(0)

recognized_this_session = set()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_SIZE)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))
        id_, conf = recognizer.predict(face)      # id_ and confidence
        # lower confidence = better match for LBPH (0 is perfect)
        if conf < 70:   # threshold adjust
            name = id_name_map.get(id_, f"ID_{id_}")
            label = f"{name} ({int(conf)})"
            if name not in recognized_this_session:
                done = mark_attendance(name)
                if done:
                    print(f"Attendance marked for {name} at {datetime.now().strftime('%H:%M:%S')}")
                recognized_this_session.add(name)
        else:
            label = f"Unknown ({int(conf)})"

        # Draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition - Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Attendance saved to", os.path.abspath(ATTENDANCE_FILE))
