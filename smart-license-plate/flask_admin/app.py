import os
import sqlite3
import csv
import subprocess
from flask import Flask, render_template, send_file, redirect, url_for, request, session, jsonify
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this in production
 
# Absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "logs", "plates.db")
EXPORT_PATH = os.path.join(BASE_DIR, "logs", "exported_logs.csv")
SNAPSHOTS_FOLDER = os.path.join(BASE_DIR, "snapshots")
 
detection_process = None
 
USERNAME = "admin"
PASSWORD = "admin123"
 
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == USERNAME and request.form.get("password") == PASSWORD:
            session["user"] = USERNAME
            return redirect("/logs")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")
 
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")
 
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect("/")
        return f(*args, **kwargs)
    return decorated
 
@app.route("/logs")
@login_required
def logs():
    query = request.args.get("query", "").strip()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if query:
        cursor.execute("""
            SELECT plate, timestamp, confidence, image_path, id
            FROM plate_logs
            WHERE plate LIKE ? OR timestamp LIKE ?
            ORDER BY id DESC LIMIT 200
        """, ('%' + query + '%', '%' + query + '%'))
    else:
        cursor.execute("""
            SELECT plate, timestamp, confidence, image_path, id
            FROM plate_logs
            ORDER BY id DESC LIMIT 200
        """)
    data = cursor.fetchall()
    conn.close()
    return render_template("logs.html", data=data, query=query)
 
@app.route("/delete/<int:log_id>", methods=["POST"])
@login_required
def delete_log(log_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM plate_logs WHERE id=?", (log_id,))
    row = cursor.fetchone()
    if row and row[0]:
        image_path = os.path.join(BASE_DIR, row[0].replace("\\", "/"))
        if os.path.exists(image_path):
            os.remove(image_path)
    cursor.execute("DELETE FROM plate_logs WHERE id=?", (log_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('logs'))
 
@app.route("/snapshots/<filename>")
def snapshots(filename):
    return send_file(os.path.join(SNAPSHOTS_FOLDER, secure_filename(filename)))
 
@app.route("/export")
@login_required
def export():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plate_logs")
    rows = cursor.fetchall()
    conn.close()
 
    with open(EXPORT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Plate", "Timestamp", "Confidence", "Image Path"])
        writer.writerows(rows)
 
    return send_file(EXPORT_PATH, as_attachment=True)
 
@app.route("/start")
@login_required
def start_detection():
    global detection_process
    if detection_process is None or detection_process.poll() is not None:
        detection_process = subprocess.Popen(["python", "main.py"], cwd=BASE_DIR)
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})
 
@app.route("/stop")
@login_required
def stop_detection():
    global detection_process
    if detection_process is not None and detection_process.poll() is None:
        detection_process.terminate()
        detection_process = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)