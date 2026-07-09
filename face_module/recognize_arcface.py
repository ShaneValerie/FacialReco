"""
recognize_arcface.py - Real-time gate recognition using InsightFace
(ArcFace 512-d embeddings, RetinaFace detector).

Same behavior as recognize.py: matches faces against enrolled
embeddings, logs attendance via the PHP API on a confirmed match,
raises an unknown-individual alert when an unenrolled face persists.

Matching uses COSINE SIMILARITY (higher = more similar), unlike the
dlib version which used Euclidean distance (lower = more similar).

Run:  py -3.11 recognize_arcface.py
Key:  Q = quit
"""

import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import mysql.connector
import requests
from insightface.app import FaceAnalysis

# ------------------------- settings ----------------------------
SIM_THRESHOLD = 0.40      # min cosine similarity to count as a match
                          # raise toward 0.5 if you see false accepts,
                          # lower toward 0.35 if real employees are missed
CONFIRM_FRAMES = 3        # consecutive matches required before logging
UNKNOWN_FRAMES = 12       # consecutive unknowns before raising the alarm
COOLDOWN_MIN = 3          # minutes before the same employee logs again
ALARM_COOLDOWN_SEC = 60   # seconds between unknown alarms
PROCESS_EVERY = 3         # process every Nth frame (speed)
CAMERA_SOURCE = 1         # your webcam index; later: "rtsp://..."
GATE_NAME = "MAIN_ENTRY"
API_BASE = "http://localhost/tenant_api"
SNAPSHOT_DIR = "snapshots"
# ----------------------------------------------------------------

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load enrolled embeddings from MySQL
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
known_ids, known_names, emb_rows = [], {}, []
for emp_id, first, last, blob in cur.fetchall():
    emb = np.frombuffer(blob, dtype=np.float32)
    if emb.shape[0] != 512:
        # Old dlib embedding still in the table - skip it
        continue
    known_ids.append(emp_id)
    known_names[emp_id] = f"{first} {last}"
    emb_rows.append(emb)
cur.close()
conn.close()

if not emb_rows:
    print("No ArcFace embeddings found. Run enroll_from_photos.py first")
    print("(and TRUNCATE the table if it still holds old dlib data).")
    raise SystemExit

known_matrix = np.vstack(emb_rows)   # shape: (num_embeddings, 512)
print(f"Loaded {len(emb_rows)} embeddings "
      f"for {len(set(known_ids))} employee(s). Starting camera...")

cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Could not open the camera source.")
    raise SystemExit

match_streak = {}
last_logged = {}
unknown_streak = 0
last_alarm = None
frame_n = 0
overlay_faces = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_n += 1

    if frame_n % PROCESS_EVERY == 0:
        overlay_faces = []
        faces = app.get(frame)
        any_unknown = False

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.normed_embedding.astype(np.float32)

            # cosine similarity against every enrolled embedding
            sims = known_matrix @ emb
            best = int(np.argmax(sims))
            best_sim = float(sims[best])

            if best_sim >= SIM_THRESHOLD:
                emp = known_ids[best]
                name = known_names[emp]
                match_streak[emp] = match_streak.get(emp, 0) + 1
                overlay_faces.append(
                    ((x1, y1, x2, y2), f"{name} {best_sim:.2f}", (0, 200, 0))
                )

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
                                "confidence": round(best_sim, 3),
                            },
                            timeout=5,
                        )
                        print(f"[{datetime.now():%H:%M:%S}] "
                              f"Attendance logged: {name} "
                              f"(sim {best_sim:.2f}) -> {r.status_code}")
                        last_logged[emp] = datetime.now()
                        match_streak[emp] = 0
                    except requests.RequestException as err:
                        print(f"API error while logging attendance: {err}")
            else:
                any_unknown = True
                overlay_faces.append(
                    ((x1, y1, x2, y2), f"UNKNOWN {best_sim:.2f}", (0, 0, 255))
                )

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

    for (x1, y1, x2, y2), label, color in overlay_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Gate Recognition ArcFace (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
