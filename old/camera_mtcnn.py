import datetime
import os
import cv2
import subprocess
import atexit
from deepface import DeepFace
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from core.tools import extract_color_features
from retinaface import RetinaFace
from mtcnn import MTCNN

tracked_faces = {}
next_id = 0
previous_frame = None
detector = MTCNN()
process = None

def cleanup_subprocess():
    if process.poll() is None:
        print("Terminating subprocess...")
        process.terminate()
        process.wait()

atexit.register(cleanup_subprocess)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray_frame
        continue

    frame_diff = cv2.absdiff(previous_frame, gray_frame)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    previous_frame = gray_frame

    if motion_score > (100 * 1000):
        print(f"Skipping frame due to high motion: {motion_score}")
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    if process == None or process.poll() is not None:
        print("RUNNING SUB PROCESS")
        process = subprocess.Popen(["python3", "validate.py"])

    detected_ids = []

    for face in faces:
        if face['confidence'] < 0.9: continue
        x, y, width, height = face['box']

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        bbox = (x1, y1, x2, y2)
        face_roi = frame[y1:y2, x1:x2]
        features = extract_color_features(frame, bbox)
        matched_id = None

        for face_id, data in tracked_faces.items():
            if data["features"] is not None: 
                similarity = 1 - cosine(features, data["features"])
                if similarity > 0.4:
                    matched_id = face_id
                    break   

        if matched_id is None:
            tracked_faces[next_id] = {
                "features": features,
                "coordinates": bbox,
                "last_seen": 0
            }

            person_dir = f"/Users/rishabh/Desktop/miniproject/FDIAS/unknown/person_{next_id}"
            if not os.path.exists(person_dir): os.makedirs(person_dir)

            # Save the frame in the person's directory with the current timestamp in milliseconds
            timestamp = int(datetime.datetime.now().timestamp() * 1000)
            frame_path = os.path.join(person_dir, f"{timestamp}.jpg")
            cv2.imwrite(frame_path, face_roi)

            matched_id = next_id
            next_id += 1

        tracked_faces[matched_id]["coordinates"] = bbox
        tracked_faces[matched_id]["last_seen"] = 0
        detected_ids.append(matched_id)

    for face_id in list(tracked_faces.keys()):
        if face_id not in detected_ids:
            tracked_faces[face_id]["last_seen"] += 1
            if tracked_faces[face_id]["last_seen"] > 10:
                del tracked_faces[face_id]

    for face_id, data in tracked_faces.items():
        (x1, y1, x2, y2) = data["coordinates"]
        label = f"Name: {face_id}" if data["features"] is None else f"ID: {face_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  break

cap.release()
cv2.destroyAllWindows()