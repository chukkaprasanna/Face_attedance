# train_model.py
import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")

os.makedirs(TRAINER_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".jpg") or f.lower().endswith(".png"):
                image_paths.append(os.path.join(root, f))

    face_samples = []
    ids = []
    for imagePath in image_paths:
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # extract id from folder name like "1-Name"
        folder = os.path.basename(os.path.dirname(imagePath))
        try:
            id_str = folder.split("-")[0]
            id_ = int(id_str)
        except:
            continue

        # if face already resized by capture script use directly; else detect and resize
        faces = face_detector.detectMultiScale(img)
        if len(faces) == 0:
            # maybe already cropped: use whole image
            face = cv2.resize(img, (200,200))
            face_samples.append(face)
            ids.append(id_)
        else:
            for (x,y,w,h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200,200))
                face_samples.append(face)
                ids.append(id_)
    return face_samples, ids

if __name__ == "__main__":
    print("Training model. This may take a few seconds...")
    faces, ids = get_images_and_labels(DATASET_DIR)
    if len(faces) == 0:
        print("No training images found. Run capture_faces.py first.")
        exit(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)
    print(f"Model trained and saved to {TRAINER_FILE}")
