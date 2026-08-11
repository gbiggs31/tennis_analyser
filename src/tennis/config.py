"""Paths and tunable constants.

Raw video and derived artifacts deliberately live *outside* the repo: a 30-minute
1080p120 clip is 8-15 GB, and the repo sits inside a OneDrive-synced folder.
Override the location with the TENNIS_DATA_ROOT environment variable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TENNIS_DATA_ROOT", r"C:\tennis_data"))

RAW_DIR = DATA_ROOT / "raw"
SESSIONS_DIR = DATA_ROOT / "sessions"
DB_PATH = DATA_ROOT / "tennis.db"

# --- Proxy video -----------------------------------------------------------
# All computer-vision work runs on this downscaled copy. Clip export cuts from
# the original so slow-motion review keeps the full 120fps.
PROXY_WIDTH = 960
PROXY_HEIGHT = 540
PROXY_FPS = 30

# The source phone tags every clip with rotation=270, but the raw sensor frame
# is already the correct landscape view - honouring the flag turns the court on
# its side. Default to ignoring it; pass --autorotate for footage from a camera
# whose flag is trustworthy.
APPLY_ROTATION_DEFAULT = False

# --- Audio -----------------------------------------------------------------
AUDIO_SAMPLE_RATE = 48_000

# Racket impact is a broadband click concentrated in this band. Wind rumble and
# most voice energy sit below it; the top end is bounded by the anti-alias limit.
BANDPASS_LOW_HZ = 700.0
BANDPASS_HIGH_HZ = 10_000.0

# 128-sample hop at 48kHz = 2.67ms resolution, far finer than one 120fps frame.
STFT_N_FFT = 1024
STFT_HOP = 128

# You physically cannot strike a ball twice faster than this.
MIN_HIT_GAP_S = 0.15

# --- Geometry / acoustics --------------------------------------------------
SPEED_OF_SOUND_MS = 343.0
NEAR_PLAYER_DISTANCE_M = 4.0
FAR_PLAYER_DISTANCE_M = 26.0

# --- Segmentation ----------------------------------------------------------
RALLY_BREAK_S = 2.5     # gap between hits that ends a rally
SERVE_LEAD_GAP_S = 3.0  # quiet gap that must precede a serve
SECOND_SERVE_MAX_GAP_S = 4.0

# --- Shot windows ----------------------------------------------------------
SHOT_PRE_S = 1.00   # clip starts this far before contact
SHOT_POST_S = 0.70  # clip ends this far after contact
SHEET_PRE_S = 0.55  # contact sheet spans this window
SHEET_POST_S = 0.30
SHEET_GRID = (3, 3)
SHEET_TILE_PX = 364          # 3 * 364 = 1092, ~1590 image tokens
SHEET_BBOX_SCALE = 2.2


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "session"


def session_id_for(video_path: Path) -> str:
    """Stable, readable id: filename stem plus a short hash of the full path.

    The hash keeps two identically-named files in different folders distinct.
    """
    digest = hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:8]
    return f"{slugify(video_path.stem)}-{digest}"


class SessionPaths:
    """Resolved artifact locations for one video."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.root = SESSIONS_DIR / session_id

    @classmethod
    def for_video(cls, video_path: Path) -> SessionPaths:
        return cls(session_id_for(Path(video_path)))

    @property
    def probe(self) -> Path:
        return self.root / "probe.json"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def proxy(self) -> Path:
        return self.root / "proxy.mp4"

    @property
    def hits(self) -> Path:
        return self.root / "hits.json"

    @property
    def points(self) -> Path:
        return self.root / "points.json"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def sheets(self) -> Path:
        return self.root / "sheets"

    @property
    def debug(self) -> Path:
        return self.root / "debug"

    def ensure(self) -> None:
        for d in (self.root, self.clips, self.sheets, self.debug):
            d.mkdir(parents=True, exist_ok=True)
