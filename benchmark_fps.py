"""
benchmark_fps.py
----------------
Experiment A-1: End-to-End FPS

• Opens the same webcam as main.py (CFG.CAMERA_INDEX).
• Runs Pipeline.process(frame) for ~60 seconds.
• Collects instantaneous FPS for every frame.
• Prints mean and stdev, then exits.

No GUI pop-up – this is headless to keep timing clean.
"""

import time
import statistics
import cv2

from config import CFG
from core.pipeline import Pipeline

DURATION_SEC = 60          # how long to sample


def main() -> None:
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CFG.CAMERA_INDEX}")

    pipeline = Pipeline()
    fps_samples = []

    start = time.perf_counter()
    prev = start

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- pipeline inference -------------------------------------------
        _ = pipeline.process(frame)

        # --- instantaneous FPS -------------------------------------------
        now = time.perf_counter()
        inst_fps = 1.0 / (now - prev)
        fps_samples.append(inst_fps)
        prev = now

        # --- stop after duration -----------------------------------------
        if now - start >= DURATION_SEC:
            break

    cap.release()

    if fps_samples:
        mean_fps = statistics.mean(fps_samples)
        stdev_fps = statistics.stdev(fps_samples) if len(fps_samples) > 1 else 0.0
        print(f"\n--- FPS Benchmark ({DURATION_SEC}s) ---")
        print(f"Frames processed : {len(fps_samples)}")
        print(f"Mean FPS         : {mean_fps:.2f}")
        print(f"Std-dev FPS      : {stdev_fps:.2f}")
    else:
        print("No frames captured – check camera.")


if __name__ == "__main__":
    main()
