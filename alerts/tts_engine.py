"""
Cross‑platform TTS helper.

• macOS       – uses the bullet‑proof `/usr/bin/say` CLI (no crashes).
• everything else – falls back to pyttsx3 (same queue interface).

Public call
-----------
>>> from alerts.tts_engine import speak
>>> speak("Wake up!")
"""

from __future__ import annotations
import os
import sys
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional

from config import CFG

# ------------------------------------------------------------------ #
# Platform detection
# ------------------------------------------------------------------ #
IS_MAC = sys.platform == "darwin"

# ------------------------------------------------------------------ #
# Shared queue for non‑blocking speak()
# ------------------------------------------------------------------ #
_MSG_Q: "queue.Queue[str]" = queue.Queue(maxsize=30)
_WORKER: Optional[threading.Thread] = None
_LOCK = threading.Lock()


# ---------------------- macOS implementation ---------------------- #
def _say_mac(text: str) -> None:
    """Fire‑and‑forget call to the system 'say' command."""
    subprocess.Popen(
        ["/usr/bin/say", text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------- pyttsx3 fallback --------------------------- #
def _say_pyttsx3(text: str) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", CFG.TTS_RATE)
    if CFG.TTS_VOICE:
        engine.setProperty("voice", CFG.TTS_VOICE)
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:  # noqa: BLE001
        print("[TTS] pyttsx3 failed:", exc)
    finally:
        engine.stop()


# ---------------------- worker loop -------------------------------- #
def _run_loop() -> None:
    while True:
        txt = _MSG_Q.get()
        if txt is None:  # sentinel for shutdown
            break

        try:
            if IS_MAC:
                _say_mac(txt)
            else:
                _say_pyttsx3(txt)
        except Exception as exc:  # noqa: BLE001
            print("[TTS] background error:", exc)

        _MSG_Q.task_done()


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER is None:
        with _LOCK:
            if _WORKER is None:
                _WORKER = threading.Thread(
                    target=_run_loop, name="TTS‑Worker", daemon=True
                )
                _WORKER.start()


# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #
def speak(text: str) -> None:
    """Non‑blocking speech; drops message if TTS disabled or queue full."""
    if not CFG.TTS_ENABLED:
        print(f"[TTS disabled] {text}")
        return

    _ensure_worker()
    try:
        _MSG_Q.put_nowait(text)
    except queue.Full:
        print("[TTS] queue full – message dropped")


def shutdown() -> None:
    """Flush remaining messages and stop the worker thread."""
    if _WORKER and _WORKER.is_alive():
        _MSG_Q.put(None)
        _WORKER.join(timeout=2)
