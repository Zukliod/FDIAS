import datetime
import os
import cv2
import subprocess
import atexit
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from core.tools import extract_color_features, is_blurry, is_far, is_moving


facemodel = YOLO('yolov11n-face.pt')
tracked_faces = {}
next_id = 0
previous_frame = None

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
    
    if previous_frame is None:
        previous_frame = frame
    elif is_moving(previous_frame, frame):
        previous_frame = frame
    else:
        results_face = facemodel.predict(frame, save=False, conf=0.5, verbose=False, iou=0.7)
        face_boxes = results_face[0].boxes.data.tolist()
        
        frame_height, frame_width = frame.shape[:2]
        margin = 20
        
        if process == None or process.poll() is not None:
            process = subprocess.Popen(["python3", "monitor.py"])
            
        detected_ids = []

        for face_box in face_boxes:
            x1, y1, x2, y2, conf = map(int, face_box[:5])
            bbox = (x1, y1, x2, y2)

            if is_far(frame, bbox):
                print("Skipping frame due to far face")
                continue

            if (not (x1 > margin and y1 > margin and x2 < frame_width - margin and y2 < frame_height - margin)):
                continue

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
                cv2.imwrite(frame_path, frame)

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