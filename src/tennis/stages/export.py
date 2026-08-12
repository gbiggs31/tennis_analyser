"""Stage 6 - the deliverable: clips of the player's shots, sorted by stroke.

Everything upstream exists to make this possible. What lands here is a folder
per stroke type containing playable clips, named so the filename alone says
which point and shot it was, how the model judged it, and how sure it was.

Two filters matter and are deliberately conservative:

* Only `event_type == "strike"`, because roughly half the detected events are
  ball bounces or net cords rather than shots.
* Only the near player by default - these are your shots.

Low-confidence labels are kept but segregated into a `review/` folder rather
than mixed in. A folder of correctly-labelled forehands is worth more than a
larger folder you cannot trust.

Near-simultaneous strikes are also collapsed. The audio detector sometimes fires
twice on one swing - racket contact and the ball leaving the strings, or a frame
of ringing - and two events 0.2s apart cannot both be shots: nobody hits twice
that fast. Left alone they become two nearly identical clips of the same shot.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import config, ffmpeg
from ..config import SessionPaths
from ..video import load_probe
from .segment import load as load_points


@dataclass
class Exported:
    custom_id: str
    stroke: str
    confidence: str
    path: Path


# Two strikes closer together than this are the same swing detected twice.
# A rally shot every 0.8-1.2s is normal; 0.35s apart is not physically possible.
MIN_SHOT_SEPARATION_S = 0.35

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _dedupe(
    selected: list[tuple[str, dict]], shots: dict
) -> tuple[list[tuple[str, dict]], int]:
    """Collapse strikes too close together to be separate shots.

    Keeps the more confident of a colliding pair, breaking ties on the later
    event - the second detection is usually the ball leaving the strings, which
    sits closer to true contact than the initial transient.
    """
    ordered = sorted(
        (c for c in selected if c[0] in shots),
        key=lambda c: shots[c[0]][1].t_contact,
    )
    kept: list[tuple[str, dict]] = []
    dropped = 0
    for cid, lab in ordered:
        t = shots[cid][1].t_contact
        if kept:
            prev_cid, prev_lab = kept[-1]
            if t - shots[prev_cid][1].t_contact < MIN_SHOT_SEPARATION_S:
                if CONFIDENCE_RANK.get(lab["confidence"], 0) >= CONFIDENCE_RANK.get(
                    prev_lab["confidence"], 0
                ):
                    kept[-1] = (cid, lab)
                dropped += 1
                continue
        kept.append((cid, lab))
    return kept, dropped


def _labels(session_id: str) -> dict[str, dict]:
    path = SessionPaths(session_id).root / "labels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No labels.json for {session_id!r}. Run `tennis label ... collect` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def summary(session_id: str, player: str = "near") -> dict:
    """What would be exported, without cutting anything. Cheap to run."""
    labels = _labels(session_id)
    strokes: Counter[str] = Counter()
    conf: Counter[str] = Counter()
    n_strike = 0
    for label in labels.values():
        if label["event_type"] != "strike" or label["player"] != player:
            continue
        n_strike += 1
        strokes[label["stroke"]] += 1
        conf[label["confidence"]] += 1
    shots = {
        f"p{p.index:03d}_s{s.index_in_point:02d}": (p, s)
        for p in load_points(session_id).points
        for s in p.shots
    }
    selected = [
        (cid, lab)
        for cid, lab in labels.items()
        if lab["event_type"] == "strike" and lab["player"] == player
    ]
    _, dropped = _dedupe(selected, shots)
    return {
        "events": len(labels),
        "strikes": n_strike,
        "duplicates_dropped": dropped,
        "clips": n_strike - dropped,
        "by_stroke": dict(strokes.most_common()),
        "by_confidence": dict(conf.most_common()),
    }


def export(
    session_id: str,
    player: str = "near",
    pre_s: float = 1.5,
    post_s: float = 1.2,
    min_confidence: str = "medium",
    slowmo: float = 1.0,
    limit: int | None = None,
) -> list[Exported]:
    paths = SessionPaths(session_id)
    probe = load_probe(session_id)
    source = Path(probe.path)
    if not source.exists():
        raise FileNotFoundError(f"Source video missing: {source}")

    labels = _labels(session_id)
    floor = CONFIDENCE_RANK.get(min_confidence, 1)

    shots = {
        f"p{p.index:03d}_s{s.index_in_point:02d}": (p, s)
        for p in load_points(session_id).points
        for s in p.shots
    }

    out_root = paths.root / "shots"
    out_root.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, dict]] = [
        (cid, lab)
        for cid, lab in sorted(labels.items())
        if lab["event_type"] == "strike" and lab["player"] == player
    ]
    selected, _ = _dedupe(selected, shots)
    if limit:
        selected = selected[:limit]

    exported: list[Exported] = []
    for cid, lab in selected:
        entry = shots.get(cid)
        if entry is None:
            continue
        point, shot = entry

        stroke = lab["stroke"] if lab["stroke"] != "none" else "unclassified"

        # A shot whose stroke never made it through the second pass still
        # carries the first pass's answer, and that pass was wrong about half
        # the time. Treat it as unverified rather than letting it sit
        # indistinguishable from a checked one.
        verified_stroke = "stroke_model" in lab
        # Uncertain calls go somewhere separate rather than polluting the
        # stroke folders, so what remains can be trusted at a glance.
        trusted = (
            CONFIDENCE_RANK.get(lab["confidence"], 0) >= floor and verified_stroke
        )
        folder = out_root / (stroke if trusted else f"review/{stroke}")
        folder.mkdir(parents=True, exist_ok=True)

        name = (
            f"{cid}_{stroke}_q{lab['quality']}"
            f"_{lab['confidence']}_t{shot.t_contact:07.1f}s.mp4"
        )
        dest = folder / name
        if not dest.exists():
            ffmpeg.cut_clip(
                source,
                dest,
                start_s=max(0.0, shot.t_contact - pre_s),
                duration_s=pre_s + post_s,
                autorotate=probe.autorotate_applied,
                mark_at_s=min(pre_s, shot.t_contact),
                slowmo=slowmo,
            )
        exported.append(Exported(cid, stroke, lab["confidence"], dest))

    _write_index(out_root / "index.csv", session_id, exported, labels, shots)
    return exported


def _write_index(
    dest: Path,
    session_id: str,
    exported: list[Exported],
    labels: dict[str, dict],
    shots: dict,
) -> None:
    """A flat index so the clips are queryable without re-reading the JSON."""
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["clip", "point", "shot", "t_contact_s", "stroke", "quality",
             "confidence", "is_serve", "notes"]
        )
        for e in exported:
            lab = labels[e.custom_id]
            point, shot = shots[e.custom_id]
            writer.writerow([
                e.path.relative_to(dest.parent).as_posix(),
                point.index, shot.index_in_point, f"{shot.t_contact:.2f}",
                lab["stroke"], lab["quality"], lab["confidence"],
                "yes" if shot.is_serve else "", lab["notes"],
            ])
