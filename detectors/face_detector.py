"""
detectors/face_detector.py
--------------------------

Single responsibility:  load **yolov11n-face.pt** once and expose
`detect(frame)`  →  list[tuple[int, int, int, int, float]]
(x1, y1, x2, y2, confidence)  in *pixel coordinates* of the *original frame*.

The wrapper hides every Ultralytics detail, so downstream code
(core/pipeline.py, etc.) stays model‑agnostic.
"""

from __future__ import annotations
import cv2
from typing import List, Tuple

from ultralytics import YOLO
from config import CFG


class FaceDetector:

    _model: YOLO | None = None

    @classmethod
    def _load_model(cls) -> YOLO:
        if cls._model is None:
            cls._model = YOLO(str(CFG.YOLO_FACE_MODEL))
        return cls._model

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    @classmethod
    def detect(
        cls, frame_bgr: "cv2.Mat", conf_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Parameters
        ----------
        frame_bgr : np.ndarray
            Original BGR frame from `cv2.VideoCapture.read()`.
        conf_threshold : float
            Minimum confidence score to keep a detection.

        Returns
        -------
        list of (x1, y1, x2, y2, conf)
            Bounding‑boxes in absolute pixel coords.
        """
        model = cls._load_model()

        # YOLO expects RGB; Ultralytics does the conversion internally,
        # so passing BGR is fine – but you can convert if you wish.
        results = model(frame_bgr, verbose=False)

        if not results:
            return []

        dets = []
        for xyxy, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()
        ):
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append((x1, y1, x2, y2, float(conf)))

        return dets


# ------------------------------------------------------------------------- #
# Small manual test
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for x1, y1, x2, y2, conf in FaceDetector.detect(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("FaceDetector demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
