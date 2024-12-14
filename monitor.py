import os
from deepface import DeepFace
from core.bot import TelegramBot

tel = TelegramBot("7388563799:AAGlUqVQyW5RCHcAoZ1jNccomX2WeAf7D64")
tel.run()

unknown_dir = './storage/unknown/'

for person_folder in os.listdir(unknown_dir):
    person_path = os.path.join(unknown_dir, person_folder)
    if os.path.isdir(person_path):
        for frame_file in os.listdir(person_path):
            if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                frame_path = os.path.join(person_path, frame_file)
                faces = DeepFace.find(
                    frame_path,
                    db_path='./storage/faces',
                    detector_backend='retinaface',
                    enforce_detection=False,
                    model_name='Facenet512',
                    distance_metric='euclidean_l2',
                    threshold=0.9,
                    silent=True
                )

                if len(faces) > 0:
                    for face in faces:
                        if(face.empty):
                            print(face)
                            tel.bot.send_photo(
                                chat_id=5334875110,
                                photo=open(frame_path, 'rb'),
                                caption=f"Intrusion Detected At Door"
                            )
                            continue

                        name = face['identity'][0].split('/')[3].upper()
                        distance = face['distance'][0]
                        threshold = face['threshold'][0]
                        print(f"Frame {frame_file} matched with: {name} with distance: {distance} and threshold: {threshold}")
                
                os.remove(frame_path)
                os.rmdir(person_path)