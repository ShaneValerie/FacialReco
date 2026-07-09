"""
recognize.py - Real-time gate recognition for the Integrated Tenant
Management System.

Watches the webcam, matches faces against enrolled embeddings, logs
attendance through the PHP API on a confirmed match, and raises an
unknown-individual alert when an unenrolled face persists.

Run:  py -3.11 recognize.py
Key:  Q = quit
"""

import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import face_recognition
import mysql.connector
import requests

# ------------------------- settings ----------------------------
THRESHOLD = 0.50          # max face distance to count as a match
CONFIRM_FRAMES = 3        # consecutive matches required before logging
UNKNOWN_FRAMES = 12       # consecutive unknowns before raising the alarm
COOLDOWN_MIN = 3          # minutes before the same employee can log again
ALARM_COOLDOWN_SEC = 60   # seconds between unknown alarms
PROCESS_EVERY = 3         # process every Nth frame (speed)
CAMERA_SOURCE = 1      # 0 = webcam; later: "rtsp://user:pass@ip:554/..."
GATE_NAME = "MAIN_ENTRY"
API_BASE = "http://localhost/tenant_api"
SNAPSHOT_DIR = "snapshots"
# ----------------------------------------------------------------

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load enrolled embeddings and names from MySQL
conn = mysql.connector.connect(
    host="localhost", user="root", password="", database="tenant_mgmt"
)
cur = conn.cursor()
cur.execute(
    """SELECT f.employee_id, e.first_name, e.last_name, f.embedding
       FROM facial_data_reference f
       JOIN employees e ON e.employee_id = f.employee_id
       WHERE e.is_active = 1"""
)
known_ids, known_names, known_encs = [], {}, []
for emp_id, first, last, blob in cur.fetchall():
    known_ids.append(emp_id)
    known_names[emp_id] = f"{first} {last}"
    known_encs.append(np.frombuffer(blob, dtype=np.float64))
cur.close()
conn.close()

print(f"Loaded {len(known_encs)} embeddings "
      f"for {len(set(known_ids))} employee(s). Starting camera...")

if len(known_encs) == 0:
    print("No enrolled faces found. Run enroll.py first.")
    raise SystemExit

cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Could not open the camera source.")
    raise SystemExit

match_streak = {}          # employee_id -> consecutive match count
last_logged = {}           # employee_id -> datetime of last attendance log
unknown_streak = 0
last_alarm = None
frame_n = 0
overlay_faces = []         # remembered boxes/labels between processed frames

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_n += 1

    if frame_n % PROCESS_EVERY == 0:
        overlay_faces = []
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        any_unknown = False

        for encoding, box in zip(encodings, boxes):
            distances = face_recognition.face_distance(known_encs, encoding)
            best = int(np.argmin(distances))
            best_dist = float(distances[best])

            # scale box back to full-size frame (we resized by 0.5)
            top, right, bottom, left = [v * 2 for v in box]

            if best_dist <= THRESHOLD:
                emp = known_ids[best]
                name = known_names[emp]
                match_streak[emp] = match_streak.get(emp, 0) + 1
                overlay_faces.append(((left, top, right, bottom),
                                      name, (0, 200, 0)))

                recently = last_logged.get(emp)
                on_cooldown = (recently is not None and
                               datetime.now() - recently
                               < timedelta(minutes=COOLDOWN_MIN))

                if match_streak[emp] >= CONFIRM_FRAMES and not on_cooldown:
                    try:
                        r = requests.post(
                            f"{API_BASE}/log_attendance.php",
                            json={
                                "employee_id": emp,
                                "gate": GATE_NAME,
                                "direction": "IN",
                                "confidence": round(1 - best_dist, 3),
                            },
                            timeout=5,
                        )
                        print(f"[{datetime.now():%H:%M:%S}] "
                              f"Attendance logged: {name} "
                              f"(conf {1 - best_dist:.2f}) -> {r.status_code}")
                        last_logged[emp] = datetime.now()
                        match_streak[emp] = 0
                    except requests.RequestException as err:
                        print(f"API error while logging attendance: {err}")
            else:
                any_unknown = True
                overlay_faces.append(((left, top, right, bottom),
                                      "UNKNOWN", (0, 0, 255)))

        if any_unknown:
            unknown_streak += 1
        else:
            unknown_streak = 0

        alarm_ready = (last_alarm is None or
                       (datetime.now() - last_alarm).total_seconds()
                       > ALARM_COOLDOWN_SEC)

        if unknown_streak >= UNKNOWN_FRAMES and alarm_ready:
            snap_path = os.path.join(
                SNAPSHOT_DIR,
                f"unknown_{datetime.now():%Y%m%d_%H%M%S}.jpg",
            )
            cv2.imwrite(snap_path, frame)
            try:
                requests.post(
                    f"{API_BASE}/report_unknown.php",
                    json={"gate": GATE_NAME, "snapshot_path": snap_path},
                    timeout=5,
                )
                print(f"[{datetime.now():%H:%M:%S}] "
                      f"ALARM: unknown individual - snapshot {snap_path}")
            except requests.RequestException as err:
                print(f"API error while reporting unknown: {err}")
            last_alarm = datetime.now()
            unknown_streak = 0

    # draw the most recent detections on every frame
    for (left, top, right, bottom), label, color in overlay_faces:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Gate Recognition (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()