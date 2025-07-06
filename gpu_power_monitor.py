#!/usr/bin/env python3
"""
gpu_power_monitor.py
--------------------
Experiment A‑4  ·  Apple‑Silicon GPU utilisation proxy
   (samples GPU power via powermetrics)

Usage
-----
sudo python gpu_power_monitor.py --seconds 60  --outfile gpu_power.csv
"""

import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PATTERN = re.compile(r"GPU Power.*?:\s*([\d.]+)\s*mW", re.I)

def run_powermetrics(duration: int):
    """
    Launch `powermetrics --samplers smc` and yield (timestamp, gpu_mw) pairs.
    """
    # -n 999999 keeps it streaming; we’ll break after duration
    proc = subprocess.Popen(
        ["powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "999999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,     # << merge stderr into the same stream
        text=True,
    )

    start = time.time()
    try:
        for line in proc.stdout:
            match = PATTERN.search(line)
            if match:
                mw = float(match.group(1))
                yield time.time() - start, mw

            if time.time() - start >= duration:
                proc.terminate()
                break
    finally:
        proc.wait(timeout=2)


def main():
    if sys.platform != "darwin" or "arm64" not in subprocess.check_output(["uname", "-m"]).decode():
        sys.exit("This test is intended for Apple‑Silicon macOS only.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60, help="sampling duration")
    parser.add_argument("--outfile", type=Path, default=Path("gpu_power.csv"))
    args = parser.parse_args()

    print(f"[INFO] Sampling GPU power for {args.seconds}s … (needs sudo)\n")
    rows = [{"sec": sec, "mw": mw} for sec, mw in run_powermetrics(args.seconds)]

    # ---------- save CSV ----------
    with args.outfile.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sec", "mw"])
        w.writeheader()
        w.writerows(rows)

    # ---------- summary ----------
    mws = [r["mw"] for r in rows]
    mean = statistics.mean(mws)
    peak = max(mws)
    p95 = statistics.quantiles(mws, n=20)[18] if len(mws) >= 20 else peak

    print(f"Samples collected : {len(rows)}")
    print(f"Mean GPU power    : {mean:6.1f} mW")
    print(f"95th‑pct          : {p95:6.1f} mW")
    print(f"Peak power        : {peak:6.1f} mW")
    print(f"CSV written to    : {args.outfile.resolve()}")


if __name__ == "__main__":
    main()
