import datetime
import os
import time
import cv2
import numpy as np
from deepface import DeepFace
from core.bot import TelegramBot

tel = TelegramBot("7388563799:AAGlUqVQyW5RCHcAoZ1jNccomX2WeAf7D64")
tel.run()

def validate_faces():
    unknown_dir = './unknown'
    for person_folder in os.listdir(unknown_dir):
        person_path = os.path.join(unknown_dir, person_folder)
        if os.path.isdir(person_path):
            for frame_file in os.listdir(person_path):
                if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                    frame_path = os.path.join(person_path, frame_file)
                    try:
                        # Load and analyze the frame
                        frame = cv2.imread(frame_path)
                        
                        # Perform face recognition
                        matches = DeepFace.find(
                            frame,
                            db_path='faces',
                            enforce_detection=False,
                            silent=True,
                            model_name='Facenet512',
                            distance_metric='euclidean_l2',
                            detector_backend='retinaface'
                        )
                        
                        if len(matches) > 0 and (not matches[0].empty):
                            matched_identity = matches[0]['identity'][0].split('/')[1].upper()
                            print(f"Frame {frame_file} matched with: {matched_identity}")


                            person_dir = os.path.join(unknown_dir, person_folder)
                            for file in os.listdir(person_dir):
                                file_path = os.path.join(person_dir, file)
                                os.remove(file_path)
                            os.rmdir(person_dir)
                        else:
                            print(f"No match found for frame {frame_file}")
                            tel.bot.send_photo(
                                chat_id=5334875110,
                                photo=open(frame_path, 'rb'),
                                caption=f"Intrusion Detected At Door"
                            )

                            person_dir = os.path.join(unknown_dir, person_folder)
                            for file in os.listdir(person_dir):
                                file_path = os.path.join(person_dir, file)
                                os.remove(file_path)
                            os.rmdir(person_dir)
                            
                    except Exception as e:
                        print(f"Error processing {frame_file}: {str(e)}")

if __name__ == "__main__":
    validate_faces()