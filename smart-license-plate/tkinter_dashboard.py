import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import Label, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk
import cv2
import sqlite3
from db_utils import DB_PATH, insert_plate_log
from camera_utils import detect_plate, save_snapshot
from log_utils import setup_logger
from mqtt_client import connect_mqtt, publish_plate
from datetime import datetime
import logging

from gpiozero import LED
from time import sleep
 
led = LED(26, active_high=False)


class LicensePlateDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 License Plate Dashboard")
        self.root.geometry("980x500")
        self.root.configure(bg="#e6f2ff")

        setup_logger()
        connect_mqtt()
        self.build_ui()

        self.cap = cv2.VideoCapture(0)
        self.last_plate = None

        self.update_frame()
        self.update_recent_logs()

    def build_ui(self):
        # Header
        header = Label(self.root, text="Smart Plate Detection", font=("Helvetica", 18, "bold"),
                       bg="#003366", fg="white", pady=6)
        header.pack(fill="x")

        # Split layout frame
        split_frame = Frame(self.root, bg="#e6f2ff")
        split_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # ========== LEFT: Video Frame ==========
        video_frame = Frame(split_frame, bg="#e6f2ff")
        video_frame.pack(side="left", padx=10, pady=5)

        self.video_label = Label(video_frame, bg="#000000", width=640, height=360)
        self.video_label.pack()

        self.info_label = Label(video_frame, text="Last Plate: None", font=("Arial", 14),
                                bg="#e6f2ff", fg="#003366", pady=10)
        self.info_label.pack()

        # ========== RIGHT: Log Frame ==========
        log_panel = Frame(split_frame, bg="#e6f2ff")
        log_panel.pack(side="right", fill="y", expand=False)

        logs_title = Label(log_panel, text="Recent Detections", font=("Arial", 14, "bold"),
                           bg="#e6f2ff", fg="#003366")
        logs_title.pack(anchor="w", padx=5, pady=(0, 5))

        log_frame_container = Frame(log_panel)
        log_frame_container.pack(fill="both", expand=True)

        canvas = Canvas(log_frame_container, width=280, height=400, bg="white", bd=2, relief="groove")
        scrollbar = Scrollbar(log_frame_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = Frame(canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.logs_canvas = canvas

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(100, self.update_frame)
            return

        plate_text, confidence, plate_img = detect_plate(frame)
        if plate_text and plate_text != self.last_plate:
            self.last_plate = plate_text
            snapshot_path = save_snapshot(plate_img, plate_text)
            insert_plate_log(plate_text, confidence, snapshot_path)
            logging.info(f"Detected: {plate_text} | Confidence: {confidence:.2f} | Snapshot: {snapshot_path}")
            self.info_label.config(text=f"Last Plate: {plate_text} | Conf: {confidence:.2f}")
            self.update_recent_logs()
             # Send MQTT
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            publish_plate(plate_text, confidence, timestamp)
            try:
                led.on()
                sleep(1.5)
                led.off()
            except:
                print("There was an exception")

        resized_frame = cv2.resize(frame, (640, 360))
        img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        self.video_label.imgtk = img
        self.video_label.configure(image=img)
        self.root.after(10, self.update_frame)

    def update_recent_logs(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT plate, timestamp FROM plate_logs ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        for plate, ts in rows:
            log_line = f"{ts} → {plate}"
            label = Label(self.scrollable_frame, text=log_line,
                          font=("Consolas", 11), anchor="w",
                          bg="white", fg="#003366")
            label.pack(fill="x", padx=10, pady=2)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = tk.Tk()
    dashboard = LicensePlateDashboard(app)
    dashboard.run()
