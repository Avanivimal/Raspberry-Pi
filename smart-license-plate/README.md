# Smart License Plate Detection & Logging System
 
A full-stack IoT system powered by Python, OpenCV, OCR, Flask, MQTT, SQLite, and ThingsBoard Cloud — all deployed on a Raspberry Pi.
 
## Features
 
- License Plate Detection: Real-time recognition using EasyOCR and OpenCV
- Local Logging: Plates, timestamps, confidence stored in SQLite
- Audible feedback: Whenever plate is detected.
- Tkinter Dashboard: View live camera feed, recent detections, and system status
- Flask Admin Panel: Remote access to logs, settings, control panel, and export options
- MQTT Integration: Publishes telemetry data to cloud brokers
- ThingsBoard Dashboard: Cloud visualization for plate entries, and logs
 
## Project Structure
 
```
smart-license-plate/
├── main.py                  # Entry point, orchestrates components
├── db_utils.py              # SQLite insert/query helpers
├── log_utils.py             # Structured logging (rotating handler)
├── mqtt_client.py           # MQTT publisher logic
├── camera_utils.py          # OpenCV image capture, snapshot logic
├── tkinter_dashboard.py     # Local GUI dashboard (Tkinter)
├── flask_admin/             # Flask web dashboard
│   ├── app.py               # Flask app with API and routes
│   └── templates/           # HTML templates for the Flask UI
│       ├── login.html       # Login page
│       └── logs.html        # Plate logs viewer
├── logs/                    # Persistent logs and database
│   ├── detection.log        # Logging output (rotating)
│   └── plates.db            # SQLite database
├── snapshots/               # Saved snapshots of detected plates
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
 
## Installation
 
### Python Dependencies
 
pip install -r requirements.txt
 
### System Dependencies
 
sudo apt update
sudo apt install tesseract-ocr libgl1
 
## Usage
 
### Start Detection Engine
 
python3 main.py
 
### Launch Flask Admin Panel
 
cd flask_admin
python3 app.py
 
Visit: http://<raspberrypi_ip>:5000/login
 
### Launch Tkinter Dashboard
 
python3 tkinter_dashboard.py
 
## MQTT Configuration
 
- Topic: pi/license_plate
- Payload Example:
 
{
  "plate": "KA01AB1234",
  "timestamp": "2025-06-17T08:30:00",
  "confidence": 0.92
}
 
## ThingsBoard Integration
 
1. Register your Pi as a device on https://thingsboard.io
 
## Authors
 
- Avani B Nair
- Ganga A S
- Reuben Vinod
