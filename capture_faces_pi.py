# capture-face.py
# Refactored Version (Clean + Structured)

import cv2
import os


# ======================================
# CONFIGURATION
# ======================================
DATASET_DIR = "dataset"
MAX_IMAGES = 100
FACE_SIZE = (200, 200)  # Resize all faces to same size
CAMERA_INDEX = 0


# ======================================
# HELPER FUNCTIONS
# ======================================

def create_user_folder(user_name):
    """Create folder for storing face images"""
    path = os.path.join(DATASET_DIR, user_name)
    os.makedirs(path, exist_ok=True)
    return path


def load_face_detector():
    """Load Haar Cascade face detector"""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def initialize_camera():
    """Start camera"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise Exception("Error: Could not open camera.")
    return cap


def save_face(face_img, save_path, count):
    """Resize and save face image"""
    face_resized = cv2.resize(face_img, FACE_SIZE)
    file_path = os.path.join(save_path, f"{count}.jpg")
    cv2.imwrite(file_path, face_resized)


# ======================================
# MAIN CAPTURE FUNCTION
# ======================================

def capture_faces():
    user_name = input("Enter Person Name: ").strip()
    if not user_name:
        print("Invalid name. Exiting.")
        return

    save_path = create_user_folder(user_name)
    face_cascade = load_face_detector()
    cap = initialize_camera()

    count = 0

    print("\n📷 Starting face capture...")
    print("Press 'q' to quit early.\n")

    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            # Crop face region
            face = frame[y:y + h, x:x + w]

            # Save face
            count += 1
            save_face(face, save_path, count)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Captured: {count}/{MAX_IMAGES}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        cv2.imshow("Face Capture", frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n✅ Face capture completed!")
    print(f"Total images saved: {count}")

    cap.release()
    cv2.destroyAllWindows()


# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    capture_faces()
