"""
face_gate.py - Real-time gate recognition for the CITIZEN system.
(Replaces recognize_arcface.py. Same InsightFace/ArcFace pipeline,
 but ALL database access now goes through the CITIZEN API with JWT auth
 - the module no longer connects to MySQL at all.)

- Embeddings are fetched from   GET  api/v1/face/embeddings.php
  and cached to embeddings_cache.npz so the gate can start even if
  the server is briefly unreachable. Refreshed every refresh_minutes.
- Confirmed matches are logged   POST api/v1/face/attendance.php
- Unknown-individual alarms      POST api/v1/face/alerts.php (multipart JPEG)

Run:  py -3.11 face_gate.py
Key:  Q = quit
"""

import base64
import os
import threading
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from api_client import CitizenAPI

# ── config ──────────────────────────────────────────────────────────────
api = CitizenAPI("config.ini")
cfg = api.cfg

GATE_ID          = int(cfg["gate"]["gate_id"])
SENSOR_ID        = cfg["gate"].get("sensor_id", "").strip() or None
CAMERA_SOURCE    = cfg["camera"]["source"]
CAMERA_SOURCE    = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE

SIM_THRESHOLD    = float(cfg["recognition"].get("sim_threshold", "0.40"))
CONFIRM_FRAMES   = int(cfg["recognition"].get("confirm_frames", "3"))
UNKNOWN_FRAMES   = int(cfg["recognition"].get("unknown_frames", "12"))
COOLDOWN_MIN     = int(cfg["recognition"].get("cooldown_min", "3"))
ALARM_COOLDOWN_S = int(cfg["recognition"].get("alarm_cooldown_sec", "60"))
PROCESS_EVERY    = int(cfg["recognition"].get("process_every", "3"))
REFRESH_MIN      = int(cfg["recognition"].get("refresh_minutes", "5"))

SNAPSHOT_DIR = "snapshots"
CACHE_FILE   = "embeddings_cache.npz"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── embeddings: API-first, cache fallback ───────────────────────────────
emb_lock = threading.Lock()
known_ids, known_names, known_matrix = [], {}, None


def fetch_embeddings():
    """Pull active embeddings from the API; fall back to local cache."""
    global known_ids, known_names, known_matrix
    try:
        r = api.get("face/embeddings.php", params={"mode": "active"})
        r.raise_for_status()
        rows = r.json()["data"]["embeddings"]

        ids, names, mats = [], {}, []
        for row in rows:
            emb = np.frombuffer(base64.b64decode(row["embedding_b64"]),
                                dtype=np.float32)
            if emb.shape[0] != 512:
                continue  # skip anything that isn't an ArcFace vector
            ids.append(int(row["user_id"]))
            names[int(row["user_id"])] = row["name"]
            mats.append(emb)

        if not mats:
            print("[emb] API returned 0 embeddings - run enroll_pending.py first.")
            return False

        with emb_lock:
            known_ids, known_names = ids, names
            known_matrix = np.vstack(mats)

        np.savez(CACHE_FILE,
                 matrix=known_matrix,
                 ids=np.array(ids, dtype=np.int64),
                 names=np.array([names[i] for i in ids], dtype=object))
        print(f"[emb] Loaded {len(mats)} embeddings for "
              f"{len(set(ids))} employee(s) from API.")
        return True

    except Exception as err:
        print(f"[emb] API fetch failed ({err}) - trying local cache...")
        if os.path.exists(CACHE_FILE):
            data = np.load(CACHE_FILE, allow_pickle=True)
            with emb_lock:
                known_matrix = data["matrix"]
                ids = data["ids"].tolist()
                nm  = data["names"].tolist()
                known_ids[:] = ids
                known_names.clear()
                known_names.update(dict(zip(ids, nm)))
            print(f"[emb] Loaded {known_matrix.shape[0]} embeddings from cache.")
            return True
        return False


def refresh_loop():
    """Background refresh so new enrollments appear without a restart."""
    while True:
        time.sleep(REFRESH_MIN * 60)
        fetch_embeddings()


# ── startup ─────────────────────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

api.login()
if not fetch_embeddings():
    print("No embeddings available (API down and no cache). Exiting.")
    raise SystemExit

threading.Thread(target=refresh_loop, daemon=True).start()

cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Could not open the camera source.")
    raise SystemExit

match_streak, last_logged = {}, {}
unknown_streak, last_alarm, frame_n = 0, None, 0
overlay_faces = []

# ── main loop ───────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_n += 1

    if frame_n % PROCESS_EVERY == 0:
        overlay_faces = []
        faces = app.get(frame)
        any_unknown = False

        with emb_lock:
            matrix = known_matrix
            ids    = list(known_ids)
            names  = dict(known_names)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.normed_embedding.astype(np.float32)

            sims = matrix @ emb
            best = int(np.argmax(sims))
            best_sim = float(sims[best])

            if best_sim >= SIM_THRESHOLD:
                uid  = ids[best]
                name = names[uid]
                match_streak[uid] = match_streak.get(uid, 0) + 1
                overlay_faces.append(
                    ((x1, y1, x2, y2), f"{name} {best_sim:.2f}", (0, 200, 0)))

                recently = last_logged.get(uid)
                on_cooldown = (recently is not None and
                               datetime.now() - recently
                               < timedelta(minutes=COOLDOWN_MIN))

                if match_streak[uid] >= CONFIRM_FRAMES and not on_cooldown:
                    try:
                        r = api.post_json("face/attendance.php", {
                            "user_id":    uid,
                            "gate_id":    GATE_ID,
                            "sensor_id":  SENSOR_ID,
                            "confidence": round(best_sim, 3),
                        })
                        body = r.json()
                        action = body.get("data", {}).get("action", "?")
                        print(f"[{datetime.now():%H:%M:%S}] "
                              f"{action.upper()}: {name} "
                              f"(sim {best_sim:.2f}) -> {r.status_code}")
                        last_logged[uid] = datetime.now()
                        match_streak[uid] = 0
                    except Exception as err:
                        print(f"API error while logging attendance: {err}")
            else:
                any_unknown = True
                overlay_faces.append(
                    ((x1, y1, x2, y2), f"UNKNOWN {best_sim:.2f}", (0, 0, 255)))

        unknown_streak = unknown_streak + 1 if any_unknown else 0

        alarm_ready = (last_alarm is None or
                       (datetime.now() - last_alarm).total_seconds()
                       > ALARM_COOLDOWN_S)

        if unknown_streak >= UNKNOWN_FRAMES and alarm_ready:
            snap_path = os.path.join(
                SNAPSHOT_DIR, f"unknown_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(snap_path, frame)          # local copy for evidence
            ok_jpg, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                r = api.post_multipart(
                    "face/alerts.php",
                    data={"gate_id": GATE_ID,
                          "sensor_id": SENSOR_ID or ""},
                    files={"snapshot": ("snapshot.jpg", buf.tobytes(),
                                        "image/jpeg")} if ok_jpg else None,
                )
                print(f"[{datetime.now():%H:%M:%S}] "
                      f"ALARM: unknown individual -> {r.status_code} "
                      f"(local copy: {snap_path})")
            except Exception as err:
                print(f"API error while reporting unknown: {err}")
            last_alarm = datetime.now()
            unknown_streak = 0

    for (x1, y1, x2, y2), label, color in overlay_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("CITIZEN Gate Recognition (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
