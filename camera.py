import datetime
from sort.tracker import SortTracker
import numpy as np
import subprocess
import cv2
import os
import atexit
from dotenv import load_dotenv
from core.tools import is_far, is_moving
from core.yolo import YOLOv8_face

load_dotenv()

facemodel = YOLOv8_face("weights/yolov8n-face.onnx", conf_thres=0.45, iou_thres=0.5)
sort = SortTracker(max_age=10)

cap = cv2.VideoCapture(1)
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

    # if frame_count % 2 == 0:
    #     continue

    if frame_count < 5: continue

    if previous_frame is not None and is_moving(previous_frame, frame):
        previous_frame = frame.copy()
    else:
        previous_frame = frame.copy()

        if process == None or process.poll() is not None:
            python_executable = os.path.join("./venv", "bin", "python") if os.name != "nt" else os.path.join("./venv", "Scripts", "python.exe")
            process = subprocess.Popen([python_executable, "monitor.py"])

        try:
            boxes, scores, classids, kpts = facemodel.detect(frame)
            detections = []

            for box, score, kp in zip(boxes, scores, kpts):
                x, y, w, h = box.astype(int)

                if is_far(frame, (x, y, (x + w), (y + h))):
                    # print("FACE TOO FAR")
                    continue

                lefteye = int(kp[0 * 3])
                righteye = int(kp[1 * 3])

                cv2.circle(frame, (int(kp[0 * 3]), int(kp[0 * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                cv2.circle(frame, (int(kp[1 * 3]), int(kp[1 * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                
                eye_threshold = 30
                eye_difference = abs(lefteye - righteye)
                
                if eye_difference <= eye_threshold:
                    # print("NOT LOOKING", eye_difference)
                    continue

                detections.append([x, y, x + w, y + h, score, 0])

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
                    cv2.imwrite(frame_path, previous_frame)

                label = f"ID: {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except:
            print("NO FACE DETECTED")
    
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
