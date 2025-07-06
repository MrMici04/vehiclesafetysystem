"""
alerts/alert_manager.py
-----------------------

• Converts raw events (emotion label, drowsiness flag) into *driver‑friendly* messages.
• Debounces each alert with a configurable cool‑down to avoid spamming.
• Sends the final text to both on‑screen overlay (returned to caller) and TTS (`speak`).
• Suppresses drowsiness alerts when active emotions (Happy, Angry) are detected.

Typical use
-----------
>>> from alerts.alert_manager import AlertManager
>>> am = AlertManager()
>>> overlay = am.handle(emotion="Angry", drowsy=False, bbox=(x1,y1,x2,y2))
>>> for text, pos, color in overlay:
...     cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
"""

from __future__ import annotations
import time
from typing import List, Tuple, Optional

from config import CFG
from alerts.tts_engine import speak


class AlertManager:
    """
    Parameters
    ----------
    cooldown_sec : float
        Minimum time between two identical alerts.
    """

    def __init__(self, cooldown_sec: float = 8.0):
        self.cooldown_sec = cooldown_sec
        self._last_sent: dict[str, float] = {}  # alert_key → timestamp
        
        # Emotions that indicate alertness and should suppress drowsiness alerts
        self.alert_emotions = {"Happy", "Angry"}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _should_fire(self, key: str) -> bool:
        """Debounce identical alerts."""
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if now - last >= self.cooldown_sec:
            self._last_sent[key] = now
            return True
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def handle(
        self,
        emotion: Optional[str],
        drowsy: bool,
        bbox: Tuple[int, int, int, int] | None = None,
    ) -> List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]]:
        """
        Decide which alerts to raise and return overlay instructions.
        
        Drowsiness alerts are suppressed when active emotions (Happy, Angry) are detected.

        Returns
        -------
        list of (text, position, bgr_color)
        """
        overlays: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []

        # 1) Emotion‑based prompts
        if emotion in CFG.EMOTION_GUIDANCE:
            msg = CFG.EMOTION_GUIDANCE[emotion]
            key = f"emotion:{emotion}"
            if self._should_fire(key):
                speak(msg)
            if bbox:
                x1, y1, x2, y2 = bbox
                overlays.append((msg, (x1, y2 + 30), (0, 255, 255)))

        # 2) Drowsiness alert (but suppress if active emotions detected)
        if drowsy:
            # Check if we should suppress drowsiness alert due to active emotion
            suppress_drowsy = emotion in self.alert_emotions
            
            if not suppress_drowsy:
                msg = "Drowsiness Alert, Wake up immediately"
                key = "drowsy"
                if self._should_fire(key):
                    speak(msg)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    overlays.append((msg, (x1, y2 + 60), (0, 0, 255)))
            else:
                # Optional: Add a subtle indicator that drowsiness was detected but suppressed
                if bbox:
                    x1, y1, x2, y2 = bbox
                    overlays.append((f"Alert suppressed ({emotion} detected)", (x1, y2 + 60), (128, 128, 128)))

        return overlays