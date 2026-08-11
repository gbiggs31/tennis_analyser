"""Video reading helpers.

Exists to enforce one invariant: **OpenCV and ffmpeg must agree on orientation.**

OpenCV honours a container's rotation flag by default, while our ffmpeg calls
deliberately override it to zero (see `ffmpeg._rotation_flags`). Left alone, the
same source video yields upright frames through one path and sideways frames
through the other, and every pixel coordinate computed from a proxy stops
matching the original. Always open captures through `open_capture`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2

from . import config
from .config import SessionPaths
from .models import ProbeInfo


def open_capture(path: Path, autorotate: bool = config.APPLY_ROTATION_DEFAULT) -> cv2.VideoCapture:
    """Open a video with orientation handling matched to the ffmpeg pipeline."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    if not autorotate:
        # Available since OpenCV 4.5; harmless where the property is unsupported.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    return cap


@contextmanager
def capture(path: Path, autorotate: bool = config.APPLY_ROTATION_DEFAULT) -> Iterator[cv2.VideoCapture]:
    cap = open_capture(path, autorotate=autorotate)
    try:
        yield cap
    finally:
        cap.release()


def load_probe(session_id: str) -> ProbeInfo:
    paths = SessionPaths(session_id)
    if not paths.probe.exists():
        raise FileNotFoundError(f"No probe.json for {session_id!r}. Run `tennis ingest` first.")
    return ProbeInfo.model_validate(json.loads(paths.probe.read_text(encoding="utf-8")))


def frame_at(cap: cv2.VideoCapture, t: float):
    """Read the frame nearest to time `t` in seconds. Returns None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None
