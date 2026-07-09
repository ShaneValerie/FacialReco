"""
enroll_from_photos.py - Batch face enrollment from photo files
Uses InsightFace (ArcFace, 512-d embeddings).

Folder layout expected:
    photos/
        1/            <- folder name = employee_id in the database
            id_photo.jpg
            selfie1.jpg
        2/
            photo.png
        ...

Each employee folder can hold 1 or more photos (jpg/jpeg/png).
Every valid photo becomes one embedding row in facial_data_reference.

Run:  py -3.11 enroll_from_photos.py
"""

import os

import cv2
import numpy as np
import mysql.connector
from insightface.app import FaceAnalysis

PHOTOS_DIR = "photos"
VALID_EXT = (".jpg", ".jpeg", ".png")
MIN_FACE_PX = 80          # reject faces smaller than this (pixels tall)
MIN_DET_SCORE = 0.60      # reject low-confidence detections

# ------------------------------------------------------------------
print("Loading ArcFace model (first run downloads ~300 MB)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

conn = mysql.connector.connect(
    host="localhost", user="root", password="", database="tenant_mgmt"
)
cur = conn.cursor()

if not os.path.isdir(PHOTOS_DIR):
    print(f"Create a '{PHOTOS_DIR}' folder next to this script first.")
    raise SystemExit

enrolled, skipped = 0, []

for folder in sorted(os.listdir(PHOTOS_DIR)):
    folder_path = os.path.join(PHOTOS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # Folder name must be a numeric employee_id
    try:
        emp_id = int(folder)
    except ValueError:
        skipped.append((folder, "folder name is not an employee ID"))
        continue

    # Employee must exist in the database
    cur.execute(
        "SELECT first_name, last_name FROM employees WHERE employee_id = %s",
        (emp_id,),
    )
    row = cur.fetchone()
    if row is None:
        skipped.append((folder, "no such employee_id in database"))
        continue

    name = f"{row[0]} {row[1]}"
    photo_count = 0

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(VALID_EXT):
            continue
        path = os.path.join(folder_path, fname)

        img = cv2.imread(path)
        if img is None:
            skipped.append((path, "could not read image file"))
            continue

        faces = app.get(img)

        if len(faces) == 0:
            skipped.append((path, "no face detected"))
            continue
        if len(faces) > 1:
            skipped.append((path, f"{len(faces)} faces found, need exactly 1"))
            continue

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (y2 - y1) < MIN_FACE_PX:
            skipped.append((path, "face too small in photo"))
            continue
        if face.det_score < MIN_DET_SCORE:
            skipped.append((path, "low detection confidence"))
            continue

        # normed_embedding is L2-normalized, 512 float32 values
        emb = face.normed_embedding.astype(np.float32)
        cur.execute(
            "INSERT INTO facial_data_reference (employee_id, embedding) "
            "VALUES (%s, %s)",
            (emp_id, emb.tobytes()),
        )
        photo_count += 1
        enrolled += 1

    conn.commit()
    print(f"Employee {emp_id} ({name}): {photo_count} photo(s) enrolled")

cur.close()
conn.close()

print(f"\nDone. {enrolled} embeddings saved.")
if skipped:
    print(f"\nSkipped {len(skipped)} item(s):")
    for item, reason in skipped:
        print(f"  - {item}: {reason}")