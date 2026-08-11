"""Locating the players in a frame.

The camera is fixed, which makes background subtraction viable and cheap. A
per-pixel median over sampled frames gives an empty-court plate; anything that
differs from it is a person, a ball or a shadow.

This exists because the two players cannot share one crop. The near player fills
several hundred pixels of height, the far player around thirty. A tile framed for
one is useless for the other, so each is located and cropped separately.

**The background must be local in time.** These are evening recordings and the
light shifts over a session: a single median plate for a 6-minute clip matched no
individual frame, 9% of the near band differed from it, and the morphological
close merged that scatter into one blob spanning the full frame width. The crops
built from it were unusable, and nothing noticed because "a blob was found" was
treated as success. Backgrounds are therefore built per 30-second window, and
every candidate blob is now checked for being person-shaped before it is trusted.
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
# The near player's silhouette is thousands of pixels; anything much smaller is
# the ball, a shadow fragment or a bird.
MIN_BLOB_FRACTION = {Player.NEAR: 0.015, Player.FAR: 0.0004}

# ...and no more than this. Anything larger is a lighting shift or a merged
# smear across the court, not a player.
MAX_BLOB_FRACTION = {Player.NEAR: 0.22, Player.FAR: 0.10}

# Widest a player may be, as a fraction of frame width. A lunging player is wide,
# a full-court smear is wider.
MAX_BLOB_WIDTH_FRACTION = 0.42

# Loosest acceptable height/width ratio. People are roughly upright even when
# stretching; a horizontal band across the court is not.
MIN_ASPECT = 0.55

# Backgrounds are rebuilt this often so they track the changing light.
BACKGROUND_WINDOW_S = 30.0
BACKGROUND_SAMPLES_PER_WINDOW = 24


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


class BackgroundModel:
    """Empty-court plates, one per time window, so they track the light."""

    def __init__(self, centres: np.ndarray, plates: np.ndarray):
        self.centres = centres          # (n,) window centre times, seconds
        self.plates = plates            # (n, h, w) uint8

    @property
    def shape(self) -> tuple[int, int]:
        return self.plates.shape[1], self.plates.shape[2]

    def at(self, t: float) -> np.ndarray:
        """Plate for the window nearest `t`."""
        return self.plates[int(np.argmin(np.abs(self.centres - t)))]

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, centres=self.centres, plates=self.plates)

    @classmethod
    def load(cls, path: Path) -> BackgroundModel:
        data = np.load(path)
        return cls(data["centres"], data["plates"])


def build_background(
    session_id: str,
    window_s: float = BACKGROUND_WINDOW_S,
    per_window: int = BACKGROUND_SAMPLES_PER_WINDOW,
) -> BackgroundModel:
    """One median plate per `window_s` of footage.

    Read sequentially and finalised window by window, so only one window's
    frames are ever held in memory - a whole session at proxy resolution would
    be gigabytes.
    """
    paths = SessionPaths(session_id)
    centres: list[float] = []
    plates: list[np.ndarray] = []

    with capture(paths.proxy, autorotate=True) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_every = max(1, int(round(window_s * fps / per_window)))

        bucket: list[np.ndarray] = []
        bucket_index = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            idx = int(t // window_s)
            if idx != bucket_index:
                if bucket:
                    centres.append((bucket_index + 0.5) * window_s)
                    plates.append(np.median(np.stack(bucket), axis=0).astype(np.uint8))
                bucket, bucket_index = [], idx
            if i % sample_every == 0:
                bucket.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            i += 1

        if bucket:
            centres.append((bucket_index + 0.5) * window_s)
            plates.append(np.median(np.stack(bucket), axis=0).astype(np.uint8))

    if not plates:
        raise RuntimeError(f"Could not read any frames from {paths.proxy}")
    return BackgroundModel(np.asarray(centres, dtype=np.float32), np.stack(plates))


def background_path(session_id: str) -> Path:
    return SessionPaths(session_id).root / "background.npz"


def load_or_build_background(session_id: str) -> BackgroundModel:
    dest = background_path(session_id)
    if dest.exists():
        try:
            return BackgroundModel.load(dest)
        except Exception:
            pass
    model = build_background(session_id)
    model.save(dest)
    return model


# Where each player is looked for when detection fails, as fractions of the
# frame. Far better than falling back to the whole court: these are generous
# boxes around where that player actually spends their time for this camera.
FALLBACK_BOX = {
    Player.NEAR: (0.16, 0.50, 0.84, 1.00),
    Player.FAR: (0.34, 0.46, 0.70, 0.63),
}


def fallback_box(which: Player, width: int, height: int) -> Box:
    x0, y0, x1, y1 = FALLBACK_BOX[which]
    return Box(int(x0 * width), int(y0 * height),
               int((x1 - x0) * width), int((y1 - y0) * height))


def find_stable(
    frames: list[tuple[float, np.ndarray]],
    model: BackgroundModel,
    which: Player,
) -> tuple[Box | None, int]:
    """Locate a player across several frames and take the median box.

    A single frame can catch the player mid-occlusion, or a shadow at just the
    wrong moment. Agreeing across a few frames either side of contact is far
    steadier, and the number of successful detections is a usable confidence
    signal in its own right.
    """
    boxes = [
        b
        for t, gray in frames
        if (b := find(gray, model.at(t), which)) is not None
    ]
    if not boxes:
        return None, 0

    def med(values: list[int]) -> int:
        return int(np.median(values))

    return (
        Box(
            med([b.x for b in boxes]),
            med([b.y for b in boxes]),
            med([b.w for b in boxes]),
            med([b.h for b in boxes]),
        ),
        len(boxes),
    )


def find(
    frame_gray: np.ndarray,
    background: np.ndarray,
    which: Player,
    threshold: int = 28,
) -> Box | None:
    """Largest *person-shaped* foreground blob in the court band, or None.

    Returning None is a real outcome, not a failure to be papered over: the
    caller falls back to a court-wide crop and reports the rate, so a broken
    background model shows up as a number instead of silently bad sheets.
    """
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

    # Consider blobs largest-first and return the first person-shaped one, rather
    # than trusting the largest blindly. A lighting shift produces a huge smear
    # that would otherwise win every time and silently yield a useless crop.
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    for idx in order:
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])

        if area < MIN_BLOB_FRACTION[which] * mask.size:
            break                                    # everything smaller too
        if area > MAX_BLOB_FRACTION[which] * mask.size:
            continue                                 # lighting change, not a player
        if bw > MAX_BLOB_WIDTH_FRACTION * w:
            continue                                 # spans too much of the court
        if bh < MIN_ASPECT * bw:
            continue                                 # a horizontal band, not a person

        return Box(
            x=int(stats[idx, cv2.CC_STAT_LEFT]),
            y=int(stats[idx, cv2.CC_STAT_TOP]) + y0,
            w=bw,
            h=bh,
        )

    return None
