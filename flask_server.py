from flask import Flask, Response
from camera import Camera
from object_detector import ObjectDetector
from streamer import Streamer
import time

print("Importing modules OK")

app = Flask(__name__)

print("Initializing camera and detector...")

camera = Camera()
detector = ObjectDetector(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel",
    
)

streamer = Streamer(camera, detector)
streamer.start()

print("Streamer started")

def gen():
    while True:
        frame = streamer.get_jpeg()
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame + b"\r\n")

@app.route("/")
def index():
    return "Flask is running"

@app.route("/video_feed")
def video_feed():
    return Response(gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting Flask server on port 5000")
    app.run(host="0.0.0.0", port=5000, threaded=False, debug=False)
