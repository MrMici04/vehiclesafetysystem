"""
benchmark_cpu_mem.py
--------------------
Experiment A‑3  ·  CPU utilisation & memory footprint

• Requires:  pip install psutil
• Captures one frame per iteration, runs Pipeline.process(frame)
• Every second logs:
      – instantaneous CPU %  (overall process, not per‑core)
      – RSS memory in MiB
• Runs for DURATION_SEC seconds, then prints mean / max / p95
  and saves raw readings to cpu_mem_stats.csv
"""

import time
import csv
from pathlib import Path

import cv2
import psutil

from config import CFG
from core.pipeline import Pipeline

DURATION_SEC = 60               # how long to sample
CSV_PATH = Path("cpu_mem_stats.csv")


def main() -> None:
    proc = psutil.Process()
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    pipe = Pipeline()

    rows = []
    start = last_sample = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pipe.process(frame)          # main work

            now = time.perf_counter()
            if now - last_sample >= 1.0:     # sample each second
                cpu = proc.cpu_percent()     # % since last call
                mem = proc.memory_info().rss / (1024 * 1024)  # MiB
                rows.append({"sec": int(now - start), "cpu": cpu, "rss_mib": mem})
                last_sample = now

            if now - start >= DURATION_SEC:
                break
    finally:
        cap.release()

    # ---------- save CSV -------------
    CSV_PATH.write_text("")
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sec", "cpu", "rss_mib"])
        w.writeheader()
        w.writerows(rows)

    # ---------- summary --------------
    cpus = [r["cpu"] for r in rows]
    mems = [r["rss_mib"] for r in rows]

    def p95(data):
        idx = int(0.95 * len(data)) - 1
        return sorted(data)[idx]

    print(f"\nCPU / RAM benchmark  ({DURATION_SEC}s,  {len(rows)} samples)")
    print(f" CPU% :  mean {sum(cpus)/len(cpus):5.1f}   "
          f"p95 {p95(cpus):5.1f}   max {max(cpus):5.1f}")
    print(f" RSS  :  mean {sum(mems)/len(mems):6.1f} MiB   "
          f"p95 {p95(mems):6.1f}   max {max(mems):6.1f} MiB")
    print(f"Raw data saved to {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
