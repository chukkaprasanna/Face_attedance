# capture_faces.py
import cv2
import os

# CONFIG
DATASET_DIR = "dataset"
CAMERA_INDEX = 0
NUM_IMAGES = 30            # images per person
MIN_SIZE = (80, 80)

# Ensure dataset dir exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise IOError("Could not load haarcascade xml file.")

def capture_for_person(person_id: int, person_name: str):
    person_folder = os.path.join(DATASET_DIR, f"{person_id}-{person_name}")
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam. Check CAMERA_INDEX.")
        return

    print(f"Starting capture for {person_name}. Press 'q' to quit early.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_SIZE)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            # save resized consistent size
            face_resized = cv2.resize(face_img, (200, 200))
            filename = os.path.join(person_folder, f"{person_id}_{count+1}.jpg")
            cv2.imwrite(filename, face_resized)
            count += 1

            # draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Images: {count}/{NUM_IMAGES}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Capture Faces - Press q to stop", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if count >= NUM_IMAGES:
            print("Captured required images.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} face images to {person_folder}")

if __name__ == "__main__":
    print("=== Face Capture ===")
    pid = input("Enter numeric person id (e.g., 1): ").strip()
    name = input("Enter person name (no spaces, use underscore): ").strip()
    if not pid.isdigit():
        print("Person id must be numeric.")
    else:
        capture_for_person(int(pid), name)
