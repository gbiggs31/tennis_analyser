"""Motion energy from the proxy video, split by court half.

Player attribution cannot come from loudness. Review of real clips showed the
loud population contains near-side shots, far-side shots, bounces and net cords
alike - distance to the microphone separates our court from the courts beside
it, and nothing finer. What *does* separate the players is that only one of them
is swinging at any given moment, and they are in different parts of the frame.

The far player occupies perhaps thirty pixels of height against the near
player's four hundred, so raw motion energy is not comparable between the two
regions. Each band is therefore normalised against its own rolling baseline: we
measure "unusual motion for this region", not absolute movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SessionPaths
from .video import capture

# Motion energy needs no detail; this is small enough to keep a 12-minute proxy
# under a couple of minutes to scan.
WORK_WIDTH = 240
WORK_HEIGHT = 135

# Fraction of frame height where the net sits for this camera position. Above is
# the far court, below is the near court.
DEFAULT_NET_Y = 0.62

# Rows above this are sky, trees and adjacent courts - excluded so a passing car
# or a neighbouring rally cannot masquerade as far-player movement.
DEFAULT_HORIZON_Y = 0.42


@dataclass
class MotionTrace:
    times: np.ndarray
    near: np.ndarray          # normalised motion energy, near half
    far: np.ndarray           # normalised motion energy, far half
    fps: float

    def energy_at(self, t: float, window_s: float = 0.20) -> tuple[float, float]:
        """Peak normalised energy in each half within +/- window_s of `t`."""
        lo = np.searchsorted(self.times, t - window_s)
        hi = np.searchsorted(self.times, t + window_s)
        if hi <= lo:
            return 0.0, 0.0
        return float(self.near[lo:hi].max()), float(self.far[lo:hi].max())


def _normalise(signal: np.ndarray, fps: float, window_s: float = 6.0) -> np.ndarray:
    """Divide by a rolling median so the two bands become comparable.

    A rolling window rather than a global one: the near player drifts in and out
    of frame, and light changes over a session.
    """
    if signal.size == 0:
        return signal
    half = max(1, int(window_s * fps / 2))
    stride = max(1, half // 4)
    anchors = np.arange(0, signal.size, stride)
    med = np.array(
        [np.median(signal[max(0, a - half) : min(signal.size, a + half)]) for a in anchors],
        dtype=np.float32,
    )
    baseline = np.interp(np.arange(signal.size), anchors, med).astype(np.float32)
    return signal / np.maximum(baseline, 1e-6)


def trace(
    session_id: str,
    net_y: float = DEFAULT_NET_Y,
    horizon_y: float = DEFAULT_HORIZON_Y,
) -> MotionTrace:
    """Scan the proxy once, returning per-frame motion energy for each court half."""
    paths = SessionPaths(session_id)
    if not paths.proxy.exists():
        raise FileNotFoundError(f"No proxy for {session_id!r}. Run `tennis ingest` first.")

    near_raw: list[float] = []
    far_raw: list[float] = []

    with capture(paths.proxy, autorotate=True) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        net_row = int(WORK_HEIGHT * net_y)
        horizon_row = int(WORK_HEIGHT * horizon_y)

        prev: np.ndarray | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.cvtColor(
                cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)

            if prev is not None:
                diff = np.abs(small - prev)
                far_raw.append(float(diff[horizon_row:net_row].mean()))
                near_raw.append(float(diff[net_row:].mean()))
            prev = small

    near = np.asarray(near_raw, dtype=np.float32)
    far = np.asarray(far_raw, dtype=np.float32)
    # Diffs describe the interval between frames n and n+1; timestamp them midway.
    times = (np.arange(near.size) + 1.5) / fps

    return MotionTrace(
        times=times,
        near=_normalise(near, fps),
        far=_normalise(far, fps),
        fps=fps,
    )


def save(tr: MotionTrace, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, times=tr.times, near=tr.near, far=tr.far, fps=tr.fps)
    return dest


def load(dest: Path) -> MotionTrace:
    data = np.load(dest)
    return MotionTrace(
        times=data["times"], near=data["near"], far=data["far"], fps=float(data["fps"])
    )
