"""
enroll_pending.py - Enroll faces from photos already in the CITIZEN system.

HR / Admin Staff already capture a profile photo per employee through
app/adminstaff/id_capture.php. That photo IS the enrollment source:

  1. GET  face/embeddings.php?mode=pending
        -> users with a profilepic but no active embedding
  2. Download each profilepic, run ArcFace, quality-check the face
  3. POST face/embeddings.php with the 512-d embedding

Run this on any machine that can reach the server (typically the gate
workstation). Run it again any time - it only processes users who are
not enrolled yet, so it's safe to schedule or run after every batch of
new employee photos.

You can also drop EXTRA photos in photos/<user_id>/*.jpg to enroll
more samples per person (better accuracy across angles), same layout
as the old enroll_from_photos.py.

Run:  py -3.11 enroll_pending.py
"""

import base64
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from api_client import CitizenAPI

MIN_FACE_PX   = 80
MIN_DET_SCORE = 0.60
EXTRA_PHOTOS  = "photos"          # optional photos/<user_id>/*.jpg
VALID_EXT     = (".jpg", ".jpeg", ".png")
TMP_DIR       = "tmp_photos"

api = CitizenAPI("config.ini")
api.login()

print("Loading ArcFace model (first run downloads ~300 MB)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def embed_image(path):
    """Return (embedding, reason). embedding is None when rejected."""
    img = cv2.imread(path)
    if img is None:
        return None, "could not read image file"
    faces = app.get(img)
    if len(faces) == 0:
        return None, "no face detected"
    if len(faces) > 1:
        return None, f"{len(faces)} faces found, need exactly 1"
    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    if (y2 - y1) < MIN_FACE_PX:
        return None, "face too small in photo"
    if face.det_score < MIN_DET_SCORE:
        return None, "low detection confidence"
    return face.normed_embedding.astype(np.float32), None


def post_embedding(user_id, emb, source, source_file):
    r = api.post_json("face/embeddings.php", {
        "user_id":       user_id,
        "embedding_b64": base64.b64encode(emb.tobytes()).decode(),
        "model":         "arcface_buffalo_l",
        "source":        source,
        "source_file":   source_file,
    })
    ok = r.status_code in (200, 201) and r.json().get("success")
    return ok, r.json().get("message", r.status_code)


# ── 1. Enroll from existing profile photos ──────────────────────────────
r = api.get("face/embeddings.php", params={"mode": "pending"})
r.raise_for_status()
pending = r.json()["data"]["pending"]
print(f"\n{len(pending)} user(s) with a profile photo but no embedding.\n")

enrolled, skipped = 0, []
os.makedirs(TMP_DIR, exist_ok=True)

for u in pending:
    uid, name, pic = int(u["user_id"]), u["name"], u["profilepic"]
    local = os.path.join(TMP_DIR, f"user_{uid}.jpg")
    try:
        api.download(pic, local)
    except Exception as err:
        skipped.append((f"{name} (#{uid})", f"download failed: {err}"))
        continue

    emb, reason = embed_image(local)
    if emb is None:
        skipped.append((f"{name} (#{uid})", reason))
        continue

    ok, msg = post_embedding(uid, emb, "profilepic", pic)
    if ok:
        enrolled += 1
        print(f"  Enrolled {name} (#{uid}) from profile photo")
    else:
        skipped.append((f"{name} (#{uid})", f"API rejected: {msg}"))

# ── 2. Optional extra photos per user (photos/<user_id>/*.jpg) ──────────
if os.path.isdir(EXTRA_PHOTOS):
    for folder in sorted(os.listdir(EXTRA_PHOTOS)):
        fpath = os.path.join(EXTRA_PHOTOS, folder)
        if not os.path.isdir(fpath):
            continue
        try:
            uid = int(folder)
        except ValueError:
            skipped.append((folder, "folder name is not a user_id"))
            continue
        for fname in sorted(os.listdir(fpath)):
            if not fname.lower().endswith(VALID_EXT):
                continue
            emb, reason = embed_image(os.path.join(fpath, fname))
            if emb is None:
                skipped.append((f"{folder}/{fname}", reason))
                continue
            ok, msg = post_embedding(uid, emb, "photo", fname)
            if ok:
                enrolled += 1
                print(f"  Enrolled extra photo {folder}/{fname}")
            else:
                skipped.append((f"{folder}/{fname}", f"API rejected: {msg}"))

print(f"\nDone. {enrolled} embedding(s) saved.")
if skipped:
    print(f"\nSkipped {len(skipped)} item(s):")
    for item, reason in skipped:
        print(f"  - {item}: {reason}")
    print("\nTip: users skipped for photo quality need a retake in "
          "id_capture.php (frontal, well lit, one face), then rerun this.")
