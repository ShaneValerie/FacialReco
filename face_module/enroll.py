"""
enroll.py - Face enrollment for the Integrated Tenant Management System

Captures 12 face samples from the webcam, converts each to a 128-d
embedding, and saves them into the facial_data_reference table.

Run:  py -3.11 enroll.py
Keys: SPACE = capture a sample, Q = quit early
"""

import cv2
import numpy as np
import face_recognition
import mysql.connector

SAMPLES_NEEDED = 12
MIN_FACE_SIZE = 150      # face box must be at least this many pixels
MIN_BLUR = 100           # Laplacian variance below this = too blurry
CAMERA_INDEX = 1        # change to 1 if you have more than one camera

# ---------------------------------------------------------------
emp_id = int(input("Enter the employee ID to enroll: "))

conn = mysql.connector.connect(
    host="localhost", user="root", password="", database="tenant_mgmt"
)
cur = conn.cursor()

# Confirm the employee exists so we don't enroll a wrong ID
cur.execute(
    "SELECT first_name, last_name FROM employees WHERE employee_id = %s",
    (emp_id,),
)
row = cur.fetchone()
if row is None:
    print(f"No employee with ID {emp_id} found. Add them in phpMyAdmin first.")
    raise SystemExit

print(f"Enrolling: {row[0]} {row[1]}")
print("Position your face in the frame. Press SPACE to capture, Q to quit.")
print("Vary your angle slightly between captures (left, right, up, down).")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Could not open webcam. Check camera permissions or CAMERA_INDEX.")
    raise SystemExit

saved = 0
status = "Press SPACE to capture"

while saved < SAMPLES_NEEDED:
    ok, frame = cap.read()
    if not ok:
        break

    display = frame.copy()
    cv2.putText(display, f"Samples: {saved}/{SAMPLES_NEEDED}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(display, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Enrollment", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord(" "):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if len(boxes) != 1:
            status = "Need exactly ONE face in frame"
            print(status)
            continue

        top, right, bottom, left = boxes[0]
        if (bottom - top) < MIN_FACE_SIZE:
            status = "Move closer to the camera"
            print(status)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < MIN_BLUR:
            status = "Too blurry - hold still"
            print(status)
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        cur.execute(
            "INSERT INTO facial_data_reference (employee_id, embedding) "
            "VALUES (%s, %s)",
            (emp_id, encoding.tobytes()),
        )
        conn.commit()
        saved += 1
        status = f"Saved sample {saved} - change your angle slightly"
        print(status)

cap.release()
cv2.destroyAllWindows()
cur.close()
conn.close()

if saved == SAMPLES_NEEDED:
    print(f"Done! {saved} samples enrolled for employee {emp_id}.")
else:
    print(f"Stopped with {saved} samples saved.")