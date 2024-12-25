import face_recognition
import cv2
import numpy as np
import os
import pickle

from core.bot import TelegramBot
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

tel = TelegramBot(os.getenv("TOKEN"))
tel.run()

unknown_dir = './storage/unknown/'
faces_dir = "storage/faces"
encoding_file = "storage/encodings.pkl"
known_face_encodings = []
known_face_names = []

if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as f:
        data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_names = data['names']
else:
    for person_dir in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person_dir)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    face_image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(face_image)[0]
                    
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_dir)
    
    with open(encoding_file, 'wb') as f:
        pickle.dump({
            'encodings': known_face_encodings,
            'names': known_face_names
        }, f)


for person_folder in os.listdir(unknown_dir):
    person_path = os.path.join(unknown_dir, person_folder)
    if os.path.isdir(person_path):
        for frame_file in os.listdir(person_path):
            if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                face_locations = []
                face_encodings = []
                face_names = []

                frame_path = os.path.join(person_path, frame_file)

                sample = cv2.imread(frame_path)
                small_frame = cv2.resize(sample, (0, 0), fx=0.25, fy=0.25)

                rgb_small_frame = np.array(small_frame[:, :, ::-1])
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn", number_of_times_to_upsample=2)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0) 
                    cv2.rectangle(sample, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(sample, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(sample, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                print(len(face_locations), face_names)

                for name in face_names:
                    if name == "Unknown":
                        cv2.imwrite("intruder.jpg", sample)

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tel.bot.send_photo(
                            chat_id=5334875110,
                            photo=open("intruder.jpg", 'rb'),
                            parse_mode="Markdown",
                            caption=f"Unauthorized Person Detected!\nTime: {current_time}\nDetected Body: {len(face_names)}\n\nPlease check the image for more details."
                        )

                        os.remove("intruder.jpg")
                        break
                
                os.remove(frame_path)
                os.rmdir(person_path)
