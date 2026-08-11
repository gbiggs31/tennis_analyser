"""Audio loading and diagnostics.

The whole pipeline rests on the racket-ball impact being audible, so before any
detection work we check that the track actually carries energy in the impact
band. A wind-ruined or silent track is worth knowing about immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import config


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sample_rate)


def bandpass(
    x: np.ndarray,
    sample_rate: int,
    low_hz: float = config.BANDPASS_LOW_HZ,
    high_hz: float = config.BANDPASS_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    filtfilt rather than lfilter: a causal filter delays the signal by a few
    milliseconds, which would bias every onset time we measure.
    """
    nyquist = sample_rate / 2.0
    high_hz = min(high_hz, nyquist * 0.99)
    if low_hz >= high_hz:
        raise ValueError(f"Invalid band {low_hz}-{high_hz}Hz for {sample_rate}Hz audio")
    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


@dataclass
class AudioDiagnostics:
    duration_s: float
    sample_rate: int
    peak: float
    rms_db: float
    band_energy_fraction: float
    clipped_fraction: float
    silent: bool
    verdict: str


def diagnose(path: Path) -> AudioDiagnostics:
    """Decide whether this track can carry the hit detector."""
    x, sample_rate = load_mono(path)
    duration = len(x) / sample_rate if sample_rate else 0.0

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    rms_db = 20.0 * np.log10(rms) if rms > 0 else -np.inf
    clipped = float(np.mean(np.abs(x) > 0.999)) if x.size else 0.0

    # Share of total power sitting in the impact band. Wind and handling noise
    # push this down; a clean court recording sits well above the floor.
    band_fraction = 0.0
    if x.size and peak > 0:
        total = float(np.sum(x.astype(np.float64) ** 2))
        if total > 0:
            band = bandpass(x, sample_rate)
            band_fraction = float(np.sum(band.astype(np.float64) ** 2) / total)

    silent = peak < 1e-4 or rms < 1e-5

    if silent:
        verdict = "SILENT - no usable audio; hit detection must fall back to motion/pose"
    elif band_fraction < 0.005:
        verdict = (
            "POOR - almost no energy in the 700Hz-10kHz impact band; likely wind-dominated"
        )
    elif band_fraction < 0.03:
        verdict = "MARGINAL - low impact-band energy; expect to tune the threshold down"
    else:
        verdict = "GOOD - impact band well represented"

    return AudioDiagnostics(
        duration_s=duration,
        sample_rate=sample_rate,
        peak=peak,
        rms_db=rms_db,
        band_energy_fraction=band_fraction,
        clipped_fraction=clipped,
        silent=silent,
        verdict=verdict,
    )
