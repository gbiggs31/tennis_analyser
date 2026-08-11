"""Locating the players in a frame.

The camera is fixed - background pixels vary by under 2/255 across a session -
which makes background subtraction both viable and cheap. A per-pixel median
over sampled frames gives a clean empty-court plate; anything that differs from
it is a person, a ball or a shadow.

This exists because the two players cannot share one crop. The near player fills
several hundred pixels of height, the far player around thirty. A tile framed for
one is useless for the other, so each is located and cropped separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SessionPaths
from .models import Player
from .video import capture

# Court occupies the lower part of the frame; above this is sky, trees, houses
# and the adjacent courts, none of which should ever be mistaken for a player.
HORIZON_Y = 0.42
NET_Y = 0.62

# A blob must cover at least this fraction of its band to count as a person.
MIN_BLOB_FRACTION = {Player.NEAR: 0.0016, Player.FAR: 0.00012}

BACKGROUND_SAMPLES = 120


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def scaled(self, factor: float, bounds: tuple[int, int]) -> Box:
        """Expand about the centre, clipped to (width, height)."""
        W, H = bounds
        w, h = self.w * factor, self.h * factor
        x, y = self.cx - w / 2, self.cy - h / 2
        x, y = max(0.0, x), max(0.0, y)
        w, h = min(w, W - x), min(h, H - y)
        return Box(int(x), int(y), int(w), int(h))

    def to_scale(self, factor: float) -> Box:
        return Box(int(self.x * factor), int(self.y * factor),
                   int(self.w * factor), int(self.h * factor))


def build_background(session_id: str, samples: int = BACKGROUND_SAMPLES) -> np.ndarray:
    """Per-pixel median of frames spread across the session: an empty court."""
    paths = SessionPaths(session_id)
    with capture(paths.proxy, autorotate=True) as cap:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // samples)
        frames = []
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(frames) >= samples:
                break
    if not frames:
        raise RuntimeError(f"Could not read any frames from {paths.proxy}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def background_path(session_id: str) -> Path:
    return SessionPaths(session_id).root / "background.png"


def load_or_build_background(session_id: str) -> np.ndarray:
    dest = background_path(session_id)
    if dest.exists():
        img = cv2.imread(str(dest), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    bg = build_background(session_id)
    cv2.imwrite(str(dest), bg)
    return bg


def find(
    frame_gray: np.ndarray,
    background: np.ndarray,
    which: Player,
    threshold: int = 28,
) -> Box | None:
    """Largest foreground blob in the given court band, or None."""
    h, w = background.shape
    band = (
        (int(h * NET_Y), h) if which == Player.NEAR else (int(h * HORIZON_Y), int(h * NET_Y))
    )
    y0, y1 = band

    diff = cv2.absdiff(frame_gray[y0:y1], background[y0:y1])
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # Close gaps between limbs and torso so a player is one blob, not five.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < MIN_BLOB_FRACTION[which] * mask.size:
        return None

    return Box(
        x=int(stats[best, cv2.CC_STAT_LEFT]),
        y=int(stats[best, cv2.CC_STAT_TOP]) + y0,
        w=int(stats[best, cv2.CC_STAT_WIDTH]),
        h=int(stats[best, cv2.CC_STAT_HEIGHT]),
    )
