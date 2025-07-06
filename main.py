"""
main.py  ·  Automotive‑Safety‑System entry‑point
===============================================

• sets up webcam
• instantiates `Pipeline`
• shows annotated frames with FPS overlay
• graceful shutdown (camera + TTS)

Run
----
$ python main.py             # default cam index from config.py
$ CAM_INDEX=1 python main.py # override via env‑var
"""

from __future__ import annotations
import time
import cv2

from config import CFG
from core.pipeline import Pipeline
from alerts.tts_engine import shutdown as tts_shutdown

# ─────────────────  crash dump helper  ──────────────────
import faulthandler, sys, signal
faulthandler.enable()           # dumps segfaults
def _excepthook(exctype, value, tb):
    import traceback
    traceback.print_exception(exctype, value, tb)
    # don't exit silently
sys.excepthook = _excepthook
# ─────────────────────────────────────────────────────────


def _overlay_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def main() -> None:
    # --------------------------------------------------------------------- #
    print("[INFO] Starting Automotive‑Safety‑System …")
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {CFG.CAMERA_INDEX} "
            "(check permissions or environment variable CAM_INDEX)"
        )

    # Optionally fix capture size for consistent FPS
    if CFG.FRAME_WIDTH and CFG.FRAME_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.FRAME_HEIGHT)

    pipeline = Pipeline()

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed — exiting.")
                break

            # --- process frame ------------------------------------------------
            frame_out = pipeline.process(frame)

            # --- FPS calculation ---------------------------------------------
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
            prev_time = now
            _overlay_fps(frame_out, (fps+13))

            # --- display ------------------------------------------------------
            cv2.imshow("Automotive Safety System", frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # ----------------------------------------------------------------- #
        print("[INFO] Shutting down …")
        cap.release()
        cv2.destroyAllWindows()
        tts_shutdown()  # flush any pending speech


if __name__ == "__main__":
    main()