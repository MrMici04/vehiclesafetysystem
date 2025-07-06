"""
benchmark_tts_latency.py
------------------------
Experiment A‑5  ·  Latency of AlertManager.handle (includes speak())

The script monkey‑patches AlertManager.handle to record the time spent
inside the function for every call that triggers a *non‑empty* message.
It stops after `MAX_SAMPLES` or `DURATION_SEC` and prints stats.

Run:
  python benchmark_tts_latency.py           # uses camera 0
"""

import time
import csv
from pathlib import Path
import functools

import cv2

from config import CFG
from core.pipeline import Pipeline
import alerts.alert_manager as alert_mod   # to patch

DURATION_SEC = 60
MAX_SAMPLES = 200
CSV_FILE = Path("tts_latency.csv")


def patch_alert_manager():
    """
    Wrap AlertManager.handle so we can time it without editing library code.
    Returns a list that will collect the latencies.
    """
    latencies = []

    orig_handle = alert_mod.AlertManager.handle

    @functools.wraps(orig_handle)
    def timed_handle(self, *args, **kwargs):
        t0 = time.perf_counter()
        res = orig_handle(self, *args, **kwargs)
        t1 = time.perf_counter()

        # Only record when a message was actually spoken / overlayed
        if res:      # overlay list not empty
            latencies.append(t1 - t0)
        return res

    alert_mod.AlertManager.handle = timed_handle
    return latencies


def main():
    latencies = patch_alert_manager()

    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    pipe = Pipeline()
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pipe.process(frame)

            if (time.perf_counter() - start >= DURATION_SEC) or (
                len(latencies) >= MAX_SAMPLES
            ):
                break
    finally:
        cap.release()

    if not latencies:
        print("No alerts fired; try making angry / surprise faces.")
        return

    # -------- save CSV ----------
    with CSV_FILE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latency_sec"])
        for x in latencies:
            w.writerow([x])

    # -------- stats --------------
    import statistics as stats

    mean = stats.mean(latencies) * 1_000  # → ms
    p95 = stats.quantiles(latencies, n=20)[18] * 1_000
    mx = max(latencies) * 1_000

    print(f"\nTTS latency over {len(latencies)} alerts")
    print(f" Mean : {mean:.3f} ms")
    print(f"  p95 : {p95:.3f} ms")
    print(f"  Max : {mx:.3f} ms")
    print(f"Raw data → {CSV_FILE.resolve()}")


if __name__ == "__main__":
    main()

