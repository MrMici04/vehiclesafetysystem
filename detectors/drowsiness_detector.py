"""
detectors/drowsiness_detector.py
--------------------------------
Compute the Eye‑Aspect‑Ratio (EAR) from dlib’s 68‑point facial landmarks
and decide whether the driver is drowsy.

Public API
----------
>>> from detectors.drowsiness_detector import DrowsinessDetector
>>> drowsy_det = DrowsinessDetector()          # one instance per driver
>>> ear, is_drowsy = drowsy_det.update(gray, bbox)
"""

from __future__ import annotations
from typing import Tuple, List

import dlib
import cv2
from scipy.spatial import distance

from config import CFG


# --------------------------------------------------------------------------- #
# Utility: eye aspect ratio
# --------------------------------------------------------------------------- #
def _ear(pts: List[Tuple[int, int]]) -> float:
    """
    pts – six landmark points ordered [p1 … p6] around the eye
    EAR = (‖p2‑p6‖ + ‖p3‑p5‖) / (2‖p1‑p4‖)
    """
    a = distance.euclidean(pts[1], pts[5])
    b = distance.euclidean(pts[2], pts[4])
    c = distance.euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c)


# --------------------------------------------------------------------------- #
# Detector class
# --------------------------------------------------------------------------- #
class DrowsinessDetector:
    """
    Keeps internal state (`closed_frames`) so short blinks don’t trigger an alert.
    Call `.update(gray_frame, face_bbox)` every frame.

    Parameters
    ----------
    ear_thresh : float
        Below this value eyes are considered “closed”.
    consec_frames : int
        How many consecutive “closed” frames before declaring drowsy.
    """

    # --- lazy shared predictor (loads once for all instances) -------------- #
    _predictor: dlib.shape_predictor | None = None

    @classmethod
    def _load_predictor(cls) -> dlib.shape_predictor:
        if cls._predictor is None:
            cls._predictor = dlib.shape_predictor(str(CFG.SHAPE_PREDICTOR))
        return cls._predictor

    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        ear_thresh: float = CFG.EAR_THRESHOLD,
        consec_frames: int = CFG.CONSEC_FRAMES_DROWSY,
    ):
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.closed_frames = 0  # running counter

    # ---------------------------------------------------------------------- #
    def update(
        self, gray: "cv2.Mat", face_bbox: Tuple[int, int, int, int]
    ) -> Tuple[float | None, bool]:
        """
        Parameters
        ----------
        gray : np.ndarray
            Grayscale frame (same size as original).
        face_bbox : (x1, y1, x2, y2)
            Face rectangle from YOLO.

        Returns
        -------
        (ear, is_drowsy)
            ear       – average EAR this frame (None if landmarks fail)
            is_drowsy – True when `closed_frames >= consec_frames`
        """
        x1, y1, x2, y2 = face_bbox
        predictor = self._load_predictor()

        try:
            shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        except RuntimeError:
            # occasionally dlib fails if bbox is out of bounds
            return None, False

        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        ear_left = _ear(left_eye)
        ear_right = _ear(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        # update counter
        if ear_avg < self.ear_thresh:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        is_drowsy = self.closed_frames >= self.consec_frames
        return ear_avg, is_drowsy


# --------------------------------------------------------------------------- #
# Quick webcam test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np

    det = DrowsinessDetector()
    cam = cv2.VideoCapture(CFG.CAMERA_INDEX)

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # naive face detector for demo (Haar)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
            gray, 1.1, 4
        )

        for (x, y, w, h) in faces[:1]:
            ear, sleepy = det.update(gray, (x, y, x + w, y + h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}" if ear else "EAR: N/A",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if sleepy:
                cv2.putText(
                    frame,
                    "Drowsiness!",
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Drowsiness demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
