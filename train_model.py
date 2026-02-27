import os
import cv2
import numpy as np

DATASET_PATH = "dataset"
MODEL_PATH = "face_model.yml"

faces = []
labels = []
label_ids = {}
current_id = 0

print("Training model...")

for root, dirs, files in os.walk(DATASET_PATH):
    for person_name in dirs:
        person_path = os.path.join(DATASET_PATH, person_name)

        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1

        label_id = label_ids[person_name]

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces.append(gray)
            labels.append(label_id)

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.save(MODEL_PATH)

np.save("labels.npy", label_ids)

print("Training completed successfully!")
print("Model saved as face_model.yml")
