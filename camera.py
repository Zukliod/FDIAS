import datetime
from sort import SortTracker as Sort
import numpy as np
from ultralytics import YOLO
import subprocess
import cv2
import os
import atexit

from core.tools import is_far, is_moving

facemodel = YOLO('./storage/yolov11n-face.pt')
sort = Sort(max_age=5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

seen_ids = set()

previous_frame = None
process = None

def cleanup_subprocess():
    if process.poll() is None:
        print("Terminating subprocess...")
        process.terminate()
        process.wait()

atexit.register(cleanup_subprocess)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count < 5: continue

    if previous_frame is not None and is_moving(previous_frame, frame):
        previous_frame = frame
    else:
        previous_frame = frame

        if process == None or process.poll() is not None:
            process = subprocess.Popen(["python3", "monitor.py"])

        results_face = facemodel.predict(source=frame, save=False, conf=0.5, verbose=False)
        face_boxes = results_face[0].boxes.data.tolist()

        frame_height, frame_width = frame.shape[:2]
        margin = 20

        detections = []
        
        for face_box in face_boxes:
            x1, y1, x2, y2, conf = map(int, face_box[:5])
            bbox = (x1, y1, x2, y2)

            if is_far(frame, bbox):
                print("Skipping frame due to far face")
                continue

            if (not (x1 > margin and y1 > margin and x2 < frame_width - margin and y2 < frame_height - margin)):
                continue

            detections.append([x1, y1, x2, y2, conf, 0])

        tracked_objects = []

        if len(detections) > 0:
            detections = np.array(detections)
            tracked_objects = sort.update(detections, frame)

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])

            if track_id not in seen_ids:
                seen_ids.add(track_id)
                person_dir = f"/Users/rishabh/Desktop/miniproject/FDIAS/storage/unknown/person_{track_id}"
                if not os.path.exists(person_dir): os.makedirs(person_dir)

                timestamp = int(datetime.datetime.now().timestamp() * 1000)
                frame_path = os.path.join(person_dir, f"{timestamp}.jpg")
                cv2.imwrite(frame_path, frame)

            label = f"ID: {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
