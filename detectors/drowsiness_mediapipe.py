from __future__ import annotations
from typing import Tuple, List
import cv2
import mediapipe as mp
from scipy.spatial import distance
from config import CFG

mp_face_mesh = mp.solutions.face_mesh

# landmark indices for left / right eye (6 points each)
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]   # like p1..p6
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

def _ear(pts: List[Tuple[float, float]]) -> float:
    a = distance.euclidean(pts[1], pts[5])
    b = distance.euclidean(pts[2], pts[4])
    c = distance.euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c)

class DrowsinessDetector:
    """MediaPipe-based EAR detector; API compatible with the old class."""

    def __init__(self,
                 ear_thresh: float = CFG.EAR_THRESHOLD,
                 consec: int = CFG.CONSEC_FRAMES_DROWSY):
        self.ear_thresh = ear_thresh
        self.consec = consec
        self.closed_frames = 0
        self.mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                          refine_landmarks=False,
                                          max_num_faces=1,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

    # keep API identical to previous version
    def update(self, bgr: "np.ndarray",
               face_bbox: Tuple[int, int, int, int]
               ) -> Tuple[float | None, bool]:
        h, w, _ = bgr.shape
        # mediapipe expects RGB full frame
        results = self.mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, False

        landmarks = results.multi_face_landmarks[0].landmark

        left = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_IDXS]
        right = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_IDXS]

        ear = (_ear(left) + _ear(right)) / 2.0

        if ear < self.ear_thresh:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        return ear, self.closed_frames >= self.consec
