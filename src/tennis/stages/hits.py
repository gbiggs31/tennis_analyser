"""Stage 1 - detect candidate ball-contact events in the audio track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config, onset
from ..audio import bandpass, load_mono
from ..config import SessionPaths
from ..models import Hit, HitsFile

# How far above the local noise floor (in MADs) the flux must rise. Tuned on the
# synthetic fixture and checked against real footage; exposed as a CLI option
# because a windy recording needs a different setting.
DEFAULT_THRESHOLD_K = 6.0

# How far a peak must rise clear of its own surroundings, also in MADs. Guards
# against noisy plateaus that sit above the threshold without containing any
# real impact.
DEFAULT_PROMINENCE_K = 3.0


@dataclass
class DetectResult:
    hits_file: HitsFile
    flux: np.ndarray
    times: np.ndarray
    threshold: np.ndarray
    sample_rate: int


def detect(
    session_id: str,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    min_gap_s: float = config.MIN_HIT_GAP_S,
    prominence_k: float = DEFAULT_PROMINENCE_K,
) -> DetectResult:
    paths = SessionPaths(session_id)
    if not paths.audio.exists():
        raise FileNotFoundError(
            f"No audio for session {session_id!r}. Run `tennis ingest` first."
        )

    x, sample_rate = load_mono(paths.audio)
    x_band = bandpass(x, sample_rate)

    flux = onset.spectral_flux(x_band)
    times = onset.frame_times(len(flux), sample_rate, config.STFT_N_FFT, config.STFT_HOP)

    frames_per_second = sample_rate / config.STFT_HOP
    median, mad = onset.rolling_baseline(flux, frames_per_second)
    floor = np.maximum(mad, 1e-6)
    threshold = median + threshold_k * floor

    peaks = onset.pick_peaks(
        flux, times, threshold, min_gap_s=min_gap_s, prominence=prominence_k * floor
    )

    envelope = onset._envelope(x_band, sample_rate)
    hits = [
        Hit(
            index=i,
            t_audio=float(times[p]),
            t_contact=float(times[p]),  # corrected once a player is attributed
            features=onset.onset_features(
                x_band, envelope, sample_rate, float(times[p]), float(flux[p])
            ),
        )
        for i, p in enumerate(peaks)
    ]

    # Split the peak-level distribution to separate events on our court from the
    # quieter background of adjacent-court play and ambience. Distance to the
    # microphone is what this measures, so it cannot tell a near-side shot from a
    # far-side one, nor a strike from a bounce - only "ours" from "not ours".
    # Player attribution happens in stage 2; event type is settled at labelling.
    own_db: float | None = None
    if len(hits) >= 4:
        own_db = onset.otsu_split(np.array([h.features.peak_db for h in hits]))
        for h in hits:
            h.is_own_court = h.features.peak_db >= own_db

    hits_file = HitsFile(
        session_id=session_id,
        audio_duration_s=len(x) / sample_rate,
        threshold_k=threshold_k,
        own_court_threshold_db=own_db,
        hits=hits,
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.hits.write_text(hits_file.model_dump_json(indent=2), encoding="utf-8")

    return DetectResult(
        hits_file=hits_file,
        flux=flux,
        times=times,
        threshold=threshold,
        sample_rate=sample_rate,
    )


def sweep(
    session_id: str,
    truth_path: Path,
    k_values: list[float],
    min_gap_s: float = config.MIN_HIT_GAP_S,
    prominence_k: float = DEFAULT_PROMINENCE_K,
) -> list[tuple[float, "Evaluation"]]:
    """Score a range of thresholds, reusing one flux computation.

    Recomputing the spectrogram per candidate threshold would dominate runtime
    on a long recording, and nothing before the threshold depends on k.
    """
    paths = SessionPaths(session_id)
    x, sample_rate = load_mono(paths.audio)
    x_band = bandpass(x, sample_rate)

    flux = onset.spectral_flux(x_band)
    times = onset.frame_times(len(flux), sample_rate, config.STFT_N_FFT, config.STFT_HOP)
    median, mad = onset.rolling_baseline(flux, sample_rate / config.STFT_HOP)
    envelope = onset._envelope(x_band, sample_rate)

    floor = np.maximum(mad, 1e-6)
    results: list[tuple[float, Evaluation]] = []
    for k in k_values:
        peaks = onset.pick_peaks(
            flux,
            times,
            median + k * floor,
            min_gap_s=min_gap_s,
            prominence=prominence_k * floor,
        )
        hf = HitsFile(
            session_id=session_id,
            audio_duration_s=len(x) / sample_rate,
            threshold_k=k,
            hits=[
                Hit(
                    index=i,
                    t_audio=float(times[p]),
                    t_contact=float(times[p]),
                    features=onset.onset_features(
                        x_band, envelope, sample_rate, float(times[p]), float(flux[p])
                    ),
                )
                for i, p in enumerate(peaks)
            ],
        )
        results.append((k, evaluate(hf, truth_path)))
    return results


def load(session_id: str) -> HitsFile:
    paths = SessionPaths(session_id)
    if not paths.hits.exists():
        raise FileNotFoundError(f"No hits.json for {session_id!r}. Run `tennis hits` first.")
    return HitsFile.model_validate_json(paths.hits.read_text(encoding="utf-8"))


# --- evaluation against a ground-truth fixture -----------------------------


@dataclass
class Evaluation:
    n_truth: int
    n_detected: int
    matched: int
    missed: list[float]
    spurious: list[float]
    bounces_detected: int
    median_error_ms: float

    @property
    def recall(self) -> float:
        return self.matched / self.n_truth if self.n_truth else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.n_detected if self.n_detected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(hits_file: HitsFile, truth_path: Path, tolerance_s: float = 0.050) -> Evaluation:
    """Match detections against known strike times.

    Bounces in the fixture are tracked separately: detecting one is not a hit
    miss, but it is a false positive the bounce classifier will need to remove.
    """
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    strikes = sorted(e["t_audio"] for e in truth["events"] if e["kind"] == "racket")
    bounces = sorted(e["t_audio"] for e in truth["events"] if e["kind"] == "bounce")
    detected = sorted(h.t_audio for h in hits_file.hits)

    unmatched = list(detected)
    matched_pairs: list[tuple[float, float]] = []
    missed: list[float] = []

    for t in strikes:
        best, best_err = None, tolerance_s
        for d in unmatched:
            err = abs(d - t)
            if err <= best_err:
                best, best_err = d, err
        if best is None:
            missed.append(t)
        else:
            matched_pairs.append((t, best))
            unmatched.remove(best)

    bounce_hits = 0
    spurious: list[float] = []
    for d in unmatched:
        if any(abs(d - b) <= tolerance_s for b in bounces):
            bounce_hits += 1
        else:
            spurious.append(d)

    errors = [abs(d - t) * 1000 for t, d in matched_pairs]
    return Evaluation(
        n_truth=len(strikes),
        n_detected=len(detected),
        matched=len(matched_pairs),
        missed=missed,
        spurious=spurious,
        bounces_detected=bounce_hits,
        median_error_ms=float(np.median(errors)) if errors else 0.0,
    )


# --- debug rendering -------------------------------------------------------


def plot_flux(
    result: DetectResult,
    dest: Path,
    truth_path: Path | None = None,
    window: tuple[float, float] | None = None,
) -> Path:
    """Flux curve, threshold and detections - the first thing to look at when
    the detector misbehaves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, t1 = window if window else (0.0, float(result.times[-1]) if len(result.times) else 1.0)
    mask = (result.times >= t0) & (result.times <= t1)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(result.times[mask], result.flux[mask], lw=0.6, color="#3b6ea5", label="spectral flux")
    ax.plot(
        result.times[mask], result.threshold[mask], lw=0.9, color="#c44e52",
        ls="--", label="adaptive threshold",
    )

    for h in result.hits_file.hits:
        if t0 <= h.t_audio <= t1:
            ax.axvline(h.t_audio, color="#55a868", lw=0.8, alpha=0.85)

    if truth_path and Path(truth_path).exists():
        truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
        for ev in truth["events"]:
            if not (t0 <= ev["t_audio"] <= t1):
                continue
            colour = "#8172b2" if ev["kind"] == "racket" else "#ccb974"
            ax.axvline(ev["t_audio"], color=colour, lw=2.0, alpha=0.35)

    ax.set_xlim(t0, t1)
    ax.set_xlabel("seconds")
    ax.set_ylabel("flux")
    ax.set_title(
        f"{result.hits_file.session_id} - {len(result.hits_file.hits)} detections "
        f"(k={result.hits_file.threshold_k})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def render_beeps(session_id: str, dest: Path, hits: list[Hit] | None = None) -> Path:
    """Mix a short tone onto the audio at every detection.

    Listening to this is by far the fastest way to judge the detector: misses
    and false positives are obvious in a way a plot never is.
    """
    import soundfile as sf

    paths = SessionPaths(session_id)
    x, sample_rate = load_mono(paths.audio)
    hits = hits if hits is not None else load(session_id).hits

    beep_len = int(0.020 * sample_rate)
    t = np.arange(beep_len) / sample_rate
    beep = (np.sin(2 * np.pi * 1800.0 * t) * np.hanning(beep_len) * 0.35).astype(np.float32)

    out = x.copy() * 0.7
    for h in hits:
        start = int(h.t_audio * sample_rate)
        stop = min(start + beep_len, len(out))
        if 0 <= start < len(out):
            out[start:stop] += beep[: stop - start]

    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out *= 0.99 / peak

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), out, sample_rate, subtype="PCM_16")
    return dest
