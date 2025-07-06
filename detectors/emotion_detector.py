"""
detectors/emotion_detector.py
-----------------------------

Wraps the FER‑style CNN stored in `face_model.h5`.

Public API
----------
>>> from detectors.emotion_detector import EmotionDetector
>>> label, conf, idx = EmotionDetector.predict(face_bgr)
"""

from __future__ import annotations
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array

from config import CFG


class EmotionDetector:

    _model: tf.keras.Model | None = None

    @classmethod
    def _load_model(cls) -> tf.keras.Model:
        if cls._model is None:
            cls._model = tf.keras.models.load_model(CFG.EMOTION_MODEL)
        return cls._model

    # ------------------------------------------------------------------ #
    # Pre‑processing util
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(face_bgr: "np.ndarray") -> np.ndarray:
        """
        Convert BGR ROI → normalized tensor shape (1, 48, 48, 1)
        expected by FER‑2013‑style networks.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Resize to 48×48
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

        # 3. Scale to [0,1]
        norm = resized.astype("float32")

        # 4. Expand dims (H,W) → (H,W,1) → (1,H,W,1)
        tensor = img_to_array(norm)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @classmethod
    def predict(
        cls, face_bgr: "np.ndarray"
    ) -> tuple[str, float, int] | tuple[None, None, None]:
        """
        Parameters
        ----------
        face_bgr : np.ndarray
            BGR image containing ONLY the face.

        Returns
        -------
        (label, confidence, index) or (None, None, None) on failure.
        """
        if face_bgr is None or min(face_bgr.shape[:2]) == 0:
            return None, None, None

        model = cls._load_model()
        tensor = cls._preprocess(face_bgr)

        preds = model.predict(tensor, verbose=0)[0]  # shape (7,)
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = CFG.EMOTION_LABELS[idx]
        return label, conf, idx


# ---------------------------------------------------------------------- #
# Small self‑test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import pathlib

    # quick test on webcam
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # naïve full‑frame test (no face crop) just to see outputs
        label, conf, _ = EmotionDetector.predict(frame)
        cv2.putText(
            frame,
            f"{label or 'N/A'} ({conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("EmotionDetector demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
