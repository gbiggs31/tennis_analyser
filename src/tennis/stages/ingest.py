"""Stage 0 - ingest.

Probes the source video, extracts a known-format audio track, and builds a small
constant-frame-rate proxy for CPU computer-vision work. Everything downstream
reads these artifacts rather than the multi-gigabyte original.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config, ffmpeg
from ..audio import AudioDiagnostics, diagnose
from ..config import SessionPaths
from ..models import ProbeInfo


@dataclass
class IngestResult:
    session_id: str
    paths: SessionPaths
    probe: ProbeInfo
    audio: AudioDiagnostics | None
    duration_mismatch_s: float | None
    warnings: list[str]


def ingest(
    video: Path,
    force: bool = False,
    autorotate: bool = config.APPLY_ROTATION_DEFAULT,
) -> IngestResult:
    video = Path(video).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"No such video: {video}")

    paths = SessionPaths.for_video(video)
    paths.ensure()
    warnings: list[str] = []

    probe = ProbeInfo(**ffmpeg.probe(video))
    probe.autorotate_applied = autorotate
    probe.proxy_width = config.PROXY_WIDTH
    probe.proxy_height = config.PROXY_HEIGHT
    probe.proxy_fps = float(config.PROXY_FPS)

    # A phone recording 120fps "slow motion" may declare a 30fps playback rate.
    # The average rate is the honest one; flag the discrepancy so the operator
    # knows which timebase the clips will carry.
    if probe.fps_nominal and probe.fps_average:
        if abs(probe.fps_nominal - probe.fps_average) > 1.0:
            warnings.append(
                f"Container reports {probe.fps_nominal:.2f}fps nominal but "
                f"{probe.fps_average:.2f}fps average - treating {probe.fps:.2f}fps as truth."
            )
    if probe.rotation and not autorotate:
        warnings.append(
            f"Ignoring the container's {probe.rotation} degree rotation flag - using the raw "
            f"sensor orientation. Re-run with --autorotate if the proxy looks sideways."
        )
    elif probe.rotation:
        warnings.append(
            f"Applying the container's {probe.rotation} degree rotation flag."
        )

    audio_diag: AudioDiagnostics | None = None
    if not probe.has_audio:
        warnings.append(
            "No audio stream. Audio-based hit detection is unavailable; "
            "Phase 1 must fall back to motion/pose."
        )
    else:
        if force or not paths.audio.exists():
            ffmpeg.extract_audio(video, paths.audio)
        audio_diag = diagnose(paths.audio)

    if force or not paths.proxy.exists():
        ffmpeg.build_proxy(video, paths.proxy, autorotate=autorotate)

    paths.probe.write_text(probe.model_dump_json(indent=2), encoding="utf-8")

    # Audio and proxy are decoded independently; if their durations disagree,
    # every timestamp downstream drifts.
    mismatch: float | None = None
    if audio_diag is not None:
        mismatch = abs(audio_diag.duration_s - probe.duration_s)
        if mismatch > 0.1:
            warnings.append(
                f"Audio duration ({audio_diag.duration_s:.3f}s) differs from video "
                f"({probe.duration_s:.3f}s) by {mismatch:.3f}s - timestamps may drift."
            )

    return IngestResult(
        session_id=paths.session_id,
        paths=paths,
        probe=probe,
        audio=audio_diag,
        duration_mismatch_s=mismatch,
        warnings=warnings,
    )
