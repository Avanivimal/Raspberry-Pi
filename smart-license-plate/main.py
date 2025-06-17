import os
import cv2
import logging
from datetime import datetime
from time import sleep
 
from camera_utils import detect_plate, save_snapshot
from db_utils import insert_plate_log, init_db
from log_utils import setup_logger
from mqtt_client import connect_mqtt, publish_plate
 
# Allow duplicate OpenMP libraries for EasyOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
# Setup
init_db()
setup_logger()
connect_mqtt()
 
 
 
# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Webcam not accessible.")
    print("Error: Cannot access the camera.")
    exit()
 
print("🔍 Press 'q' to quit the detection window.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read frame from webcam.")
        break
 
    # Plate detection
    plate_text, confidence, plate_img = detect_plate(frame)
 
    if plate_text:
        # Save image
        snapshot_path = save_snapshot(plate_img, plate_text)
 
        # Store in DB
        insert_plate_log(plate_text, confidence, snapshot_path)
 
        # Log to file and console
        logging.info(f"Detected: {plate_text} | Confidence: {confidence:.2f} | Snapshot: {snapshot_path}")
        print(f"[LOGGED] {plate_text} @ {snapshot_path}")
 
        # Send MQTT
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        publish_plate(plate_text, confidence, timestamp)
 
        # Annotate detected plate
        cv2.putText(frame, f"{plate_text} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Display feed
    cv2.imshow("License Plate Detection", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Session ended by user.")
        break
 
# Cleanup
cap.release()
cv2.destroyAllWindows()