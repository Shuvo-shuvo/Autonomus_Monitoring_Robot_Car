import threading
import time
import cv2

class Streamer:
    def __init__(self, camera, detector):
        self.camera = camera
        self.detector = detector
        self.frame = None
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            frame = self.camera.read()
            frame = self.detector.detect(frame)  # detector returns FRAME
            self.frame = frame
            time.sleep(0.01)

    def get_jpeg(self):
        if self.frame is None:
            return None
        ret, jpeg = cv2.imencode(".jpg", self.frame)
        return jpeg.tobytes() if ret else None

