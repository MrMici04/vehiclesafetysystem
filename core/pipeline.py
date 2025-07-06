"""
core/pipeline.py
----------------
FaceDetector  →  EmotionDetector  →  DrowsinessDetector  →  AlertManager
"""

from __future__ import annotations
from typing import Tuple, List

import cv2
import numpy as np

from config import CFG
from detectors.face_detector import FaceDetector
from detectors.emotion_detector import EmotionDetector
from detectors.drowsiness_detector import DrowsinessDetector
from alerts.alert_manager import AlertManager


# ---------- tunables (could also live in config.py) ------------------------ #
WARMUP_FRAMES: int = 10        # skip emotion / TTS for the first N frames
CONF_THRESHOLD: float = 0.40   # min prob to accept an emotion prediction


class Pipeline:

    def __init__(self):
        self.drowsy = DrowsinessDetector()
        self.alert_mgr = AlertManager()
        self.frame_count: int = 0            # increments every process() call

    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_face_box(
        frame: "np.ndarray", bbox: Tuple[int, int, int, int], conf: float
    ) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    @staticmethod
    def _draw_overlays(
        frame: "np.ndarray",
        overlays: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
    ) -> None:
        for text, pos, color in overlays:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ------------------------------------------------------------------ #
    def process(self, frame: "np.ndarray") -> "np.ndarray":
        """
        Parameters
        ----------
        frame : np.ndarray (BGR)

        Returns
        -------
        Annotated frame (BGR).
        """
        self.frame_count += 1
        warmup = self.frame_count <= WARMUP_FRAMES

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 1. FACE DETECTION ---------------------------------------------
        detections = FaceDetector.detect(frame)

        for (x1, y1, x2, y2, conf) in detections:
            bbox = (x1, y1, x2, y2)
            self._draw_face_box(frame, bbox, conf)

            # -------- 2. EMOTION DETECTION (skip in warm‑up) ---------------
            emotion_label = None
            if not warmup:
                face_roi_bgr = frame[y1:y2, x1:x2]
                emotion_label, emo_conf, _ = EmotionDetector.predict(face_roi_bgr)

                # apply confidence gate
                if emotion_label and emo_conf >= CONF_THRESHOLD:
                    cv2.putText(
                        frame,
                        f"{emotion_label} ({emo_conf:.2f})",
                        (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    emotion_label = None  # treat as uncertain

            # -------- 3. DROWSINESS DETECTION ------------------------------
            ear, is_drowsy = self.drowsy.update(gray, bbox)
            if ear is not None:
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # -------- 4. ALERT MANAGEMENT -----------------------------------
            overlays = self.alert_mgr.handle(emotion_label, is_drowsy, bbox)
            self._draw_overlays(frame, overlays)

        return frame


# --------------------------------------------------------------------------- #
# Minimal self‑test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    pipe = Pipeline()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = pipe.process(frame)
            cv2.imshow("Automotive Safety System", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
