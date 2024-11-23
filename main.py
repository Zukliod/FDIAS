import cv2
from ultralytics import YOLO
from deepface import DeepFace
import os

# Paths
input_video_path = 'input/4.mp4'
output_video_path = 'output/4.mp4'
database_path = 'database/'  # Folder containing images of known people
family_members = ["family_member_1.jpg", "family_member_2.jpg"]  # Add your family members' images

# Load YOLO Models
model_person = YOLO("yolov8n.pt")
model_face = YOLO("yolov8n-face.pt")

# Open video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Video propertiespip
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for .mp4 files
    fps,
    (frame_width, frame_height)
)

# Function to recognize face using DeepFace
def recognize_face(face_image):
    try:
        # Save the detected face as a temporary image
        temp_face_path = 'temp_face.jpg'
        cv2.imwrite(temp_face_path, face_image)

        # Use DeepFace for face recognition
        result = DeepFace.find(temp_face_path, db_path=database_path, enforce_detection=False)
        
        # Check if the recognized face is a family member
        if len(result) > 0:
            identity = result[0]['identity']
            if os.path.basename(identity) in family_members:
                return os.path.basename(identity)
            else:
                print(f"Intruder Alert! Unknown person detected: {identity}")
                return "Intruder"
        else:
            print("Intruder Alert! No match found.")
            return "Unknown"
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return "Unknown"

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Person detection
        results_person = model_person.predict(source=frame, save=False, conf=0.5)
        for result in results_person[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            if int(class_id) == 0:  # Class ID for 'person'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person: {confidence:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Face detection
        results_face = model_face.predict(source=frame, save=False, conf=0.5)
        for result in results_face[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            face_image = frame[int(y1):int(y2), int(x1):int(x2)]  # Extract the face region
            
            # Perform face recognition with DeepFace
            identity = recognize_face(face_image)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{identity}: {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write output frame
        out.write(frame)

        # Show frame
        cv2.imshow('Detection with Face Recognition', frame)

    except Exception as e:
        print(f"Error processing frame: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {output_video_path}")
