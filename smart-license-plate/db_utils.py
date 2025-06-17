import sqlite3
from datetime import datetime

DB_PATH = "logs/plates.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plate_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_plate_log(plate, confidence, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO plate_logs (plate, timestamp, confidence, image_path)
        VALUES (?, ?, ?, ?)
    """, (plate, timestamp, confidence, image_path))
    conn.commit()
    conn.close()
