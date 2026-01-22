import cv2
from collections import defaultdict

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class ObjectDetector:
    def __init__(self, proto, model):
        self.net = cv2.dnn.readNetFromCaffe(proto, model)

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        counts = defaultdict(int)

        # ---------- DETECTION LOOP ----------
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES):
                continue

            label = CLASSES[idx]
            counts[label] += 1

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            # BLUE BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # GREEN LABEL
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # ---------- DRAW COUNTING HUD ----------
        y = 35
        cv2.rectangle(frame, (10, 10), (260, 10 + 30 * len(counts)), (0, 0, 0), -1)

        for label, count in counts.items():
            cv2.putText(
                frame,
                f"{label.upper()}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),  # YELLOW COUNT
                2,
                cv2.LINE_AA
            )
            y += 30

        return frame

