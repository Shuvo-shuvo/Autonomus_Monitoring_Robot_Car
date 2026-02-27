import cv2
import numpy as np

MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.npy"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

label_ids = np.load(LABELS_PATH, allow_pickle=True).item()
id_labels = {v: k for k, v in label_ids.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Starting Vision Pipeline... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(face)

        if confidence < 80:
            name = id_labels[id_]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame,
                    f"{name} ({round(confidence,1)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imshow("Vision Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
