"""
config.py  ·  Centralised settings for the Automotive‑Safety‑System project
--------------------------------------------------------------------------

All hard‑coded paths, magic numbers, and tweakable thresholds live here.
Every other module imports *only* from `config` so that global settings
can be changed in one place (or via environment variables).

Usage
-----
>>> from config import CFG
>>> print(CFG.EMOTION_LABELS)
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Helper – resolve project root no matter where the script is run from.
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class _Config:
    # ---------------- Paths -------------------------------------------------- #
    # All models live in ./models/  (feel free to change)
    MODELS_DIR: Path = ROOT_DIR / "models"

    YOLO_FACE_MODEL: Path = field(
        default_factory=lambda: _ensure(
            ROOT_DIR / "models" / "yolov11n-face.pt", env="YOLO_FACE_MODEL"
        )
    )
    EMOTION_MODEL: Path = field(
        default_factory=lambda: _ensure(
            ROOT_DIR / "models" / "face_model.h5", env="EMOTION_MODEL"
        )
    )
    SHAPE_PREDICTOR: Path = field(
        default_factory=lambda: _ensure(
            ROOT_DIR / "models" / "shape_predictor_68_face_landmarks.dat",
            env="SHAPE_PREDICTOR",
        )
    )

    # ----------------‑ Camera / frame settings ------------------------------ #
    CAMERA_INDEX: int = int(os.getenv("CAM_INDEX", 0))
    FRAME_WIDTH: int = 480   # resize for speed; None keeps native
    FRAME_HEIGHT: int = 360

    # ----------------‑ Emotion detection ------------------------------------ #
    EMOTION_LABELS: tuple[str, ...] = (
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    )
    # Guidance messages when a specific emotion is detected
    EMOTION_GUIDANCE: dict[str, str] = field(
        default_factory=lambda: {
            "Angry": "Take a quick break and breathe deeply to relax.",
            "Surprise": "Emergency detected – are you alright?",
            "Sad": "Focus on the drive, you'll be alright.",
        }
    )

    # ----------------‑ Drowsiness detection --------------------------------- #
    EAR_THRESHOLD: float = 0.3       # eye closed if EAR < threshold
    CONSEC_FRAMES_DROWSY: int = 30    # frames below threshold ⇒ drowsy alert

    # ----------------‑ Text‑to‑Speech (pyttsx3) ----------------------------- #
    TTS_ENABLED: bool = True
    TTS_RATE: int = 175               # words per minute
    TTS_VOICE: str | None = None      # system default; set UUID/name to override


def _ensure(path: Path, *, env: str) -> Path:
    """
    Ensure a file exists (or an env‑override is provided). Will *not* raise here;
    existence is checked lazily when the model is first loaded so unit tests
    without real models still import `config.py`.
    """
    override = os.getenv(env)
    if override:
        return Path(override).expanduser().resolve()
    return path.expanduser().resolve()


# Export a singleton for convenience
CFG = _Config()

# --------------------------------------------------------------------------- #
# Optional: pretty‑print config if run directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pprint

    pprint.pp(CFG.__dict__, sort_dicts=False)
