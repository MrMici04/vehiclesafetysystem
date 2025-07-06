"""
benchmark_stage_latency.py
--------------------------
Experiment A-2  ·  Per-stage latency breakdown

Captures webcam frames and, for each frame:
   t0 ─ face detect ─ t1 ─ emotion CNN ─ t2 ─ EAR ─ t3 ─ alerts ─ t4

Δface   = t1-t0   (YOLO forward)
Δemo    = t2-t1   (Keras forward)
Δear    = t3-t2   (dlib predictor + EAR maths)
Δalert  = t4-t3   (alert manager & optional TTS queue)
Δtotal  = t4-t0   (whole pipeline)

Results:
• CSV  : stage_timings.csv      (one row per frame)
• STDOUT summary table
"""

import csv
import time
import statistics
import cv2

from config import CFG
from core.pipeline import Pipeline
from detectors.face_detector import FaceDetector
from detectors.emotion_detector import EmotionDetector
from detectors.drowsiness_detector import DrowsinessDetector
from alerts.alert_manager import AlertManager

DURATION_SEC = 30            # run long enough for good stats
CSV_NAME = "stage_timings.csv"


def main() -> None:
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    # manual construction of sub-modules so we can time them separately
    face_det = FaceDetector
    emo_det = EmotionDetector
    drow_det = DrowsinessDetector()
    alert_mgr = AlertManager()

    rows = []
    start_run = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        bboxes = face_det.detect(frame)
        t1 = time.perf_counter()

        # just run emotion on first face (driver) for deterministic timing
        if bboxes:
            x1, y1, x2, y2, _ = bboxes[0]
            face_roi = frame[y1:y2, x1:x2]
            emo_label, emo_conf, _ = emo_det.predict(face_roi)
        else:
            emo_label, emo_conf = None, None
        t2 = time.perf_counter()

        if bboxes:
            ear, sleepy = drow_det.update(gray, (x1, y1, x2, y2))
        else:
            ear, sleepy = None, False
        t3 = time.perf_counter()

        alert_mgr.handle(emo_label, sleepy, bboxes[0][:4] if bboxes else None)
        t4 = time.perf_counter()

        rows.append(
            {
                "face": t1 - t0,
                "emo": t2 - t1,
                "ear": t3 - t2,
                "alert": t4 - t3,
                "total": t4 - t0,
            }
        )

        # stop after DURATION_SEC
        if t4 - start_run >= DURATION_SEC:
            break

    cap.release()

    # ---------- save CSV ----------
    with open(CSV_NAME, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---------- print summary -----
    def summary(key: str):
        data = [r[key] for r in rows]
        return (
            statistics.mean(data),
            statistics.quantiles(data, n=20)[18],  # 95th percentile
            max(data),
        )

    print(f"\nLatency summary over {len(rows)} frames  (seconds)")
    print(" Stage    mean     p95     max")
    for k in ("face", "emo", "ear", "alert", "total"):
        m, p95, mx = summary(k)
        print(f"{k:>6} : {m:7.4f}  {p95:7.4f}  {mx:7.4f}")

    print(f"\nRaw data written to {CSV_NAME}")


if __name__ == "__main__":
    main()
