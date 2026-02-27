import cv2
import os

DATASET_PATH = "dataset"
USER_NAME = input("Enter Person Name: ")
SAVE_PATH = os.path.join(DATASET_PATH, USER_NAME)

os.makedirs(SAVE_PATH, exist_ok=True)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 100  

print("Starting face capture... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        count += 1

        face = frame[y:y+h, x:x+w]

        file_name = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Image {count}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= MAX_IMAGES:
        break

print("Face capture completed!")

cap.release()
cv2.destroyAllWindows()

