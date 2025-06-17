import cv2
import numpy as np
import easyocr
import imutils
import datetime
import os
import re

#from gpiozero import LED
#from time import sleep
 
#led = LED(26, active_high=False)
# Initialize OCR reader
reader = easyocr.Reader(['en'])
regex_pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,}[0-9]{1,}$'
 
 
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    return edged
 
def detect_plate(frame):
    edged = preprocess_image(frame)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
 
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = frame[y:y + h, x:x + w]
            if plate_img.size == 0:
                continue
 
            result = reader.readtext(plate_img)
            if result:
                raw_text, conf = result[0][1], result[0][2]
 
                # Clean and search for valid substrings
                cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                matches = re.findall(regex_pattern, cleaned)
                if matches:
                    # Return first valid-looking plate match
                    return matches[0], conf, plate_img
 
    return None, None, None #No valid plate found
 
def save_snapshot(image, plate_text):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{plate_text}_{timestamp}.jpg".replace(" ", "_").replace("/", "-")
    folder = "snapshots"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    '''try:
        led.on()
        sleep(1.5)
        led.off()
    except:
        print("There was an exception")'''
    return path
 