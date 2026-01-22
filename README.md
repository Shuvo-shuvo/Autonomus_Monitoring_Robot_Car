
# 🚗 Raspberry Pi Real-Time Computer Vision System  
**Object Detection, Counting & Live Flask Streaming**

---

## 📌 Overview

This project is a **real-time computer vision system** implemented on a **Raspberry Pi**, designed to detect and count objects using a deep-learning model and stream the processed video live to a web browser using **Flask (MJPEG streaming)**.

The system is optimized for **headless operation**, meaning:
- No HDMI display is required
- No `cv2.imshow()` GUI dependency
- Video is accessed remotely via a browser

This architecture is suitable for:
- Autonomous / remote-controlled cars
- Surveillance and monitoring systems
- Edge-AI and IoT computer vision projects
- Raspberry Pi-based robotics applications

---

## ✨ Key Features

- 🎯 Real-time object detection
- 🔢 Live object counting per frame
- 📷 Raspberry Pi Camera (IMX219) support
- 🌐 Live MJPEG video streaming via Flask
- 🖥️ Viewable from any browser on the same network
- ⚡ Optimized for Raspberry Pi 5 (Bookworm OS)
- 🧠 Pre-trained deep-learning model (no training required)
- 🔌 Headless operation (SSH / VNC friendly)

---

## 🧠 Technologies Used

| Category | Technology |
|--------|-----------|
| Programming Language | Python 3 |
| Computer Vision | OpenCV (DNN module) |
| Deep Learning Model | MobileNet SSD (Caffe) |
| Camera Stack | libcamera / Picamera2 |
| Streaming | Flask (MJPEG over HTTP) |
| Operating System | Raspberry Pi OS Bookworm |

---

## 🤖 Deep Learning Model Details

### MobileNet SSD (Single Shot Detector)

**Why MobileNet SSD?**
- Lightweight architecture
- Fast inference on low-power devices
- Suitable for real-time performance on Raspberry Pi
- Pre-trained on common object classes

### Model Files

```

models/
├── MobileNetSSD_deploy.prototxt
└── MobileNetSSD_deploy.caffemodel

```

### Supported Object Classes (Examples)

- person  
- car  
- chair  
- diningtable  
- dog  
- cat  
- bottle  
- bus  
- motorbike  
- sofa  
- tvmonitor  

*(Based on the standard MobileNet SSD class set)*

---

## 🏗️ System Architecture

### Processing Pipeline

```

Raspberry Pi Camera
↓
Frame Capture (camera.py)
↓
Object Detection (OpenCV DNN)
↓
Bounding Boxes + Labels + Confidence
↓
Object Counting (HUD overlay)
↓
JPEG Encoding
↓
Flask MJPEG Server
↓
Web Browser (Laptop / Phone)

```

### Architecture Highlights

- Vision processing runs entirely on the Raspberry Pi (edge computing)
- Flask only handles frame streaming (no heavy computation)
- Browser acts as a lightweight client
- Stable, low-latency design

---

## 📁 Project Structure

```

car_project/
├── camera.py              # Raspberry Pi camera interface
├── object_detector.py     # Object detection + counting logic
├── streamer.py            # Threaded frame processing
├── flask_server.py        # Flask MJPEG streaming server
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── dataset/               # (Optional) face datasets
├── capture_faces_pi.py    # (Optional) face capture
├── train_model.py         # (Optional) LBPH training
├── vision_pipeline.py    # (Optional) extended pipeline
└── README.md

````

---

## 🛠️ Hardware Requirements

- Raspberry Pi 5 (recommended)
- Raspberry Pi Camera Module (IMX219)
- Stable 5V power supply (⚠ insufficient power causes crashes)
- Laptop / PC / Phone on the same network

---

## 💻 Software Requirements

- Raspberry Pi OS **Bookworm**
- Python **3.10 or newer**
- OpenCV with DNN support
- Flask

---

## ⚙️ Installation & Setup (Ready-to-Copy)

### 1️⃣ Update System

```bash
sudo apt update && sudo apt upgrade -y
````

### 2️⃣ Install OpenCV and Required Tools

```bash
sudo apt install python3-opencv python3-pip -y
```

### 3️⃣ Install Flask

```bash
pip3 install flask
```

---

## 📷 Camera Verification

Before running the project, ensure the camera is working:

```bash
rpicam-hello
```

A live preview confirms correct camera setup.

---

## ▶️ Running the Project

Navigate to the project directory:

```bash
cd ~/car_project
```

Start the Flask server:

```bash
python3 flask_server.py
```

Expected output:

```
Running on http://0.0.0.0:5000
```

---

## 🌐 Viewing the Live Stream

Open a browser on any device connected to the same network and visit:

```
http://<RASPBERRY_PI_IP>:5000/video_feed
```

Example:

```
http://192.168.0.103:5000/video_feed
```

---

## 🎨 Output Visualization Guide

| Visual Element | Description               |
| -------------- | ------------------------- |
| 🔵 Blue Boxes  | Detected object regions   |
| 🟢 Green Text  | Object label + confidence |
| 🟨 Yellow HUD  | Object count per frame    |

Example HUD:

```
CAR: 4
CHAIR: 3
DININGTABLE: 1
```

---

## ⚠️ Common Issues & Solutions

### Camera crashes or display flickers

* Use a stable power supply
* Avoid undervoltage

### Flask page loads but no video

* Ensure `/video_feed` endpoint is used
* Verify Raspberry Pi IP address
* Refresh browser

### Low FPS

* Reduce camera resolution
* Close unnecessary background services

---

## 🚀 Future Enhancements

* Face recognition (LBPH)
* Object tracking with unique IDs
* Person-only analytics
* REST API for object counts
* Autonomous motor control logic
* Web dashboard with charts
* TensorFlow Lite optimization

---

## 📜 License

This project is released as **open-source** for educational and research purposes.

---

## 🙌 Acknowledgements

* OpenCV
* Flask
* Raspberry Pi Foundation
* MobileNet SSD
