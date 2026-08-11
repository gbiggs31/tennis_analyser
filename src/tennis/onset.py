"""Onset detection: find racket-ball impacts in the audio track.

A racket strike is a broadband click - energy appearing suddenly across many
frequencies at once. Spectral flux (the sum of positive frame-to-frame changes
in the magnitude spectrum) is the classic detector for exactly that shape, and
it runs in seconds on a CPU.

Two details matter for accuracy:

* The threshold is *local*. Court noise, traffic and crowd volume drift over a
  match, so a fixed threshold either misses quiet rallies or floods on loud
  ones. We track a rolling median and MAD instead.
* The window is short. A 128-sample hop at 48kHz gives 2.7ms resolution, which
  is a third of a frame at 120fps - fine enough that the contact frame we later
  extract is the real one.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from . import config
from .models import OnsetFeatures

# Log compression makes the detector responsive to quiet onsets (a far-player
# strike is ~16dB down) without letting loud ones dominate the flux sum.
LOG_GAMMA = 100.0


def noise_profile(
    frames: np.ndarray,
    window: np.ndarray,
    max_samples: int = 4000,
) -> np.ndarray:
    """Per-bin median magnitude: an estimate of the stationary noise floor.

    Court ambience, traffic hum and wind are roughly stationary, so their
    spectrum shows up as a persistent per-bin level. Subtracting it before
    differencing is what lets a far-player strike - 16dB down and otherwise
    level with a court bounce - clear the threshold at all.
    """
    n_frames = frames.shape[0]
    step = max(1, n_frames // max_samples)
    sample = frames[::step] * window
    mag = np.abs(np.fft.rfft(sample, axis=1)).astype(np.float32)
    return np.median(mag, axis=0)


def spectral_flux(
    x: np.ndarray,
    n_fft: int = config.STFT_N_FFT,
    hop: int = config.STFT_HOP,
    chunk_frames: int = 8192,
    whiten: bool = True,
    smooth_frames: int = 3,
) -> np.ndarray:
    """Half-wave-rectified spectral flux, one value per hop.

    Computed in chunks: a full magnitude spectrogram of a long match would be
    hundreds of megabytes, and we only ever need the 1-D flux curve.
    """
    if len(x) < n_fft:
        return np.zeros(0, dtype=np.float32)

    window = signal.get_window("hann", n_fft).astype(np.float32)
    frames = sliding_window_view(x, n_fft)[::hop]
    n_frames = frames.shape[0]

    floor = noise_profile(frames, window) if whiten else None

    flux = np.zeros(n_frames, dtype=np.float32)
    prev_mag: np.ndarray | None = None

    for start in range(0, n_frames, chunk_frames):
        stop = min(start + chunk_frames, n_frames)
        block = frames[start:stop] * window
        mag = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)
        if floor is not None:
            mag = np.maximum(mag - floor, 0.0)
        mag = np.log1p(LOG_GAMMA * mag)

        if prev_mag is not None:
            first_diff = mag[0] - prev_mag
            flux[start] = np.sum(np.maximum(first_diff, 0.0))
        diffs = np.diff(mag, axis=0)
        flux[start + 1 : stop] = np.sum(np.maximum(diffs, 0.0), axis=1)
        prev_mag = mag[-1]

    flux[0] = 0.0

    # A few frames of smoothing removes the ripple that would otherwise turn one
    # onset into a cluster of local maxima, without blurring a 2.7ms-resolution
    # transient enough to matter.
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        flux = np.convolve(flux, kernel, mode="same").astype(np.float32)

    return flux


def frame_times(n_frames: int, sample_rate: int, n_fft: int, hop: int) -> np.ndarray:
    """Timestamp of each flux frame, at the centre of its analysis window."""
    return (np.arange(n_frames) * hop + n_fft / 2.0) / sample_rate


def rolling_baseline(
    values: np.ndarray,
    frames_per_second: float,
    window_s: float = 2.0,
    stride_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Local median and MAD, evaluated on a coarse grid and interpolated.

    A true rolling median over every frame is wasteful - the noise floor moves
    on the scale of seconds, not milliseconds - so we sample it every 250ms and
    interpolate between.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    half = max(1, int(window_s * frames_per_second / 2))
    stride = max(1, int(stride_s * frames_per_second))
    anchors = np.arange(0, n, stride)

    med = np.empty(len(anchors), dtype=np.float32)
    mad = np.empty(len(anchors), dtype=np.float32)
    for i, centre in enumerate(anchors):
        lo, hi = max(0, centre - half), min(n, centre + half)
        segment = values[lo:hi]
        m = float(np.median(segment))
        med[i] = m
        mad[i] = float(np.median(np.abs(segment - m)))

    idx = np.arange(n)
    return (
        np.interp(idx, anchors, med).astype(np.float32),
        np.interp(idx, anchors, mad).astype(np.float32),
    )


def pick_peaks(
    flux: np.ndarray,
    times: np.ndarray,
    threshold: np.ndarray,
    min_gap_s: float = config.MIN_HIT_GAP_S,
    prominence: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of peaks above threshold, thinned to one per min_gap_s.

    `prominence` is what stops a noisy plateau from yielding dozens of
    detections: a real impact rises clear of its surroundings, whereas noise
    ripple wanders around the same level. scipy keeps the tallest peak within
    each `distance` window, so close-together candidates resolve to the true
    impact rather than its ringing.
    """
    if len(flux) < 3 or len(times) < 2:
        return np.zeros(0, dtype=int)

    frames_per_second = 1.0 / float(times[1] - times[0])
    distance = max(1, int(round(min_gap_s * frames_per_second)))

    peaks, _ = signal.find_peaks(
        flux,
        height=threshold,
        distance=distance,
        prominence=prominence,
    )
    return peaks.astype(int)


def otsu_split(values: np.ndarray, bins: int = 64) -> float:
    """Threshold that best separates a set of values into two groups.

    Applied to peak level in dB, this cleanly isolates near-player strikes from
    the quieter background of adjacent-court play and ambient noise. It is
    chosen per session rather than hard-coded because recording gain varies
    between clips, so any absolute dB cut would be wrong on the next video.

    Otsu's method: pick the split maximising between-class variance.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("-inf")

    counts, edges = np.histogram(finite, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return float("-inf")

    weight_lo = np.cumsum(counts) / total
    weight_hi = 1.0 - weight_lo
    csum = np.cumsum(counts * centres)
    mean_lo = np.divide(csum, np.cumsum(counts), out=np.zeros(bins), where=np.cumsum(counts) > 0)
    total_sum = csum[-1]
    remaining = counts[::-1].cumsum()[::-1]
    mean_hi = np.divide(
        total_sum - csum, remaining, out=np.zeros(bins), where=remaining > 0
    )

    between = weight_lo * weight_hi * (mean_lo - mean_hi) ** 2
    between[~np.isfinite(between)] = 0.0
    return float(edges[int(np.argmax(between)) + 1])


def _envelope(x: np.ndarray, sample_rate: int, cutoff_hz: float = 400.0) -> np.ndarray:
    """Rectified, low-passed amplitude envelope."""
    sos = signal.butter(2, cutoff_hz / (sample_rate / 2), btype="low", output="sos")
    return signal.sosfiltfilt(sos, np.abs(x)).astype(np.float32)


def onset_features(
    x_band: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    t: float,
    flux_value: float,
    pre_s: float = 0.008,
    post_s: float = 0.040,
) -> OnsetFeatures:
    """Acoustic descriptors that separate racket strikes from court bounces.

    A strike is stiff and bright: it reaches full amplitude in a millisecond or
    two and carries strong high-frequency content. A ball landing on the court
    is softer and duller on both counts.
    """
    n = len(x_band)
    start = max(0, int((t - pre_s) * sample_rate))
    stop = min(n, int((t + post_s) * sample_rate))
    if stop - start < 8:
        return OnsetFeatures(
            peak_db=-120.0,
            spectral_centroid_hz=0.0,
            attack_time_ms=0.0,
            hf_lf_ratio=0.0,
            onset_strength=float(flux_value),
        )

    segment = x_band[start:stop]
    env = envelope[start:stop]

    peak = float(np.max(np.abs(segment)))
    peak_db = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0

    # Attack time: 10% -> 90% of the envelope peak, measured up to the peak.
    peak_idx = int(np.argmax(env))
    attack_ms = 0.0
    if peak_idx > 0:
        env_peak = float(env[peak_idx])
        if env_peak > 1e-9:
            rise = env[: peak_idx + 1]
            lo_hits = np.flatnonzero(rise >= 0.1 * env_peak)
            hi_hits = np.flatnonzero(rise >= 0.9 * env_peak)
            if lo_hits.size and hi_hits.size:
                attack_ms = max(0.0, (hi_hits[0] - lo_hits[0]) / sample_rate * 1000.0)

    spectrum = np.abs(np.fft.rfft(segment * signal.get_window("hann", len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sample_rate)
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total) if total > 0 else 0.0

    lf = float(np.sum(spectrum[freqs < 2000.0]))
    hf = float(np.sum(spectrum[freqs > 4000.0]))
    hf_lf = hf / lf if lf > 1e-9 else 0.0

    return OnsetFeatures(
        peak_db=peak_db,
        spectral_centroid_hz=centroid,
        attack_time_ms=attack_ms,
        hf_lf_ratio=hf_lf,
        onset_strength=float(flux_value),
    )
