"""Stage 4 (partial) - export a reviewable video clip per detected event.

Pulled forward from its place in the pipeline because human review of real clips
is the only way to settle which detections are racket strikes and which are
near-side ball bounces. Acoustic features alone do not separate them.

Each clip carries context either side of the impact rather than the instant
itself: a strike is only recognisable from the swing that precedes it and the
follow-through after.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .. import config, ffmpeg
from ..config import SessionPaths
from ..models import Hit
from ..video import load_probe
from .hits import load as load_hits


@dataclass
class ClipSpec:
    hit: Hit
    path: Path
    start_s: float
    duration_s: float
    mark_at_s: float


def export(
    session_id: str,
    own_court_only: bool = True,
    pre_s: float = 1.5,
    post_s: float = 1.2,
    slowmo: float = 1.0,
    limit: int | None = None,
    subdir: str = "review",
) -> list[ClipSpec]:
    paths = SessionPaths(session_id)
    probe = load_probe(session_id)
    source = Path(probe.path)
    if not source.exists():
        raise FileNotFoundError(f"Source video missing: {source}")

    hits_file = load_hits(session_id)
    events = hits_file.own_court if own_court_only else hits_file.hits
    if limit:
        events = events[:limit]

    out_dir = paths.clips / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[ClipSpec] = []
    for h in events:
        start = max(0.0, h.t_contact - pre_s)
        # Near the very start of a video the lead-in is clipped, so the marker
        # position has to be measured from the actual clip start.
        mark_at = h.t_contact - start
        name = f"h{h.index:03d}_t{h.t_contact:07.2f}s_{h.features.peak_db:.0f}dB.mp4"
        dest = out_dir / name

        ffmpeg.cut_clip(
            source,
            dest,
            start_s=start,
            duration_s=pre_s + post_s,
            autorotate=probe.autorotate_applied,
            mark_at_s=mark_at,
            slowmo=slowmo,
        )
        specs.append(
            ClipSpec(hit=h, path=dest, start_s=start, duration_s=pre_s + post_s,
                     mark_at_s=mark_at)
        )

    _write_review_sheet(out_dir / "review.csv", specs)
    return specs


def export_point(
    session_id: str,
    point_index: int,
    pre_s: float = 3.0,
    post_s: float = 3.0,
    slowmo: float = 1.0,
) -> Path:
    """Export one whole point as a single continuous clip.

    Useful for judging segmentation itself rather than individual detections:
    an implausibly long "point" is usually a warm-up rally that never paused
    long enough to split.
    """
    from .segment import load as load_points

    paths = SessionPaths(session_id)
    probe = load_probe(session_id)
    source = Path(probe.path)

    points = load_points(session_id).points
    match = next((p for p in points if p.index == point_index), None)
    if match is None:
        raise IndexError(f"No point {point_index} in {session_id!r} ({len(points)} points)")

    start = max(0.0, match.t_start - pre_s)
    duration = (match.t_end - match.t_start) + pre_s + post_s

    out_dir = paths.clips / "points"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (
        f"p{match.index:03d}_{match.t_start:07.1f}s_"
        f"{match.t_end - match.t_start:.0f}s_{len(match.shots)}events.mp4"
    )

    ffmpeg.cut_clip(
        source, dest,
        start_s=start,
        duration_s=duration,
        autorotate=probe.autorotate_applied,
        slowmo=slowmo,
    )
    return dest


def _write_review_sheet(dest: Path, specs: list[ClipSpec]) -> None:
    """A CSV to fill in while watching, so verdicts come back in a form the
    detector can actually be scored against."""
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["clip", "hit_index", "t_contact_s", "peak_db",
             "verdict (strike/bounce/nothing)", "notes"]
        )
        for s in specs:
            writer.writerow(
                [s.path.name, s.hit.index, f"{s.hit.t_contact:.3f}",
                 f"{s.hit.features.peak_db:.1f}", "", ""]
            )
