# camera.py
from picamera2 import Picamera2
import time

class Camera:
    def __init__(self):
        print("Initializing camera...")
        self.picam = Picamera2()
        config = self.picam.create_video_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.picam.configure(config)
        self.picam.start()
        time.sleep(1)
        print("Camera started")

    def read(self):
        return self.picam.capture_array()
