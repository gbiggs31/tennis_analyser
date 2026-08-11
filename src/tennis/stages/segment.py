"""Stage 2 - group events into points, attribute them to a player, find serves.

Grouping is the reliable part. In match play a point is bounded by silence: the
players walk, retrieve the ball and prepare, which leaves a gap far longer than
anything inside a rally. Clustering on that gap recovers point boundaries
without needing to understand a single frame.

Attribution is **not** done here, and the reason is worth recording.

Loudness cannot do it: the loud population mixes near-side shots, far-side
shots, bounces and net cords (see the `motion` module header). The obvious
fallback, whole-court-half motion energy, was implemented and measured against
five hand-verified near-player strikes - and got all five backwards. Each half
has to be normalised against its own baseline to be comparable at all, but the
near player's constant movement inflates his own baseline until a swing barely
registers, while the near-static far half spikes at any movement, including
players on the adjacent court. The normalisation destroys the signal.

Doing it properly needs per-player localisation: background-subtract (the camera
is fixed), find each player's blob, and measure motion inside their box. That is
real work, and unnecessary - a vision model reading the contact sheet at
labelling time can see which player is swinging, and is already being asked
which stroke it was. Attribution therefore moves to stage 5, at no extra cost.

`use_motion=True` still runs the old path for experimentation. It is off by
default because it is worse than useless: it is confidently wrong.
"""

from __future__ import annotations

import numpy as np

from .. import config, motion
from ..config import SessionPaths
from ..models import Hit, Player, Point, PointsFile, Shot
from .hits import load as load_hits

# A rally that is this short and closely followed by another is a service fault:
# serve, net or long, pause, second serve.
MAX_FAULT_EVENTS = 2
MAX_FAULT_GAP_S = 12.0

# Below this margin between the two halves' motion energy, attribution is a
# coin toss and the player is left unknown rather than guessed.
MIN_ATTRIBUTION_MARGIN = 0.15


def _cluster(events: list[Hit], gap_s: float) -> list[list[Hit]]:
    """Split a time-ordered event list wherever the gap exceeds `gap_s`."""
    if not events:
        return []
    groups: list[list[Hit]] = [[events[0]]]
    for prev, cur in zip(events, events[1:]):
        if cur.t_audio - prev.t_audio > gap_s:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def _merge_faults(groups: list[list[Hit]]) -> list[tuple[list[Hit], bool]]:
    """Fold a fault and its second serve into one point.

    Returns (events, had_fault) pairs.
    """
    merged: list[tuple[list[Hit], bool]] = []
    pending: list[Hit] = []

    for i, group in enumerate(groups):
        nxt = groups[i + 1] if i + 1 < len(groups) else None
        looks_like_fault = (
            len(group) <= MAX_FAULT_EVENTS
            and nxt is not None
            and nxt[0].t_audio - group[-1].t_audio <= MAX_FAULT_GAP_S
        )
        if looks_like_fault:
            pending.extend(group)
            continue
        merged.append((pending + group, bool(pending)))
        pending = []

    if pending:
        merged.append((pending, True))
    return merged


def _attribute(hit: Hit, tr: motion.MotionTrace) -> tuple[Player, float]:
    """Assign a player from which court half moved, with a confidence in [0,1]."""
    near_e, far_e = tr.energy_at(hit.t_audio)
    total = near_e + far_e
    if total <= 1e-6:
        return Player.UNKNOWN, 0.0

    margin = abs(near_e - far_e) / total
    if margin < MIN_ATTRIBUTION_MARGIN:
        return Player.UNKNOWN, margin
    return (Player.NEAR if near_e > far_e else Player.FAR), margin


def _delay_corrected(t_audio: float, player: Player) -> float:
    """Undo sound propagation so the timestamp is the moment of contact.

    The far player is ~26m away, so their strike is heard ~76ms late - nine
    frames at 120fps, enough to miss the contact entirely when extracting it.
    """
    distance = {
        Player.NEAR: config.NEAR_PLAYER_DISTANCE_M,
        Player.FAR: config.FAR_PLAYER_DISTANCE_M,
    }.get(player)
    if distance is None:
        # Unknown player: assume mid-court rather than bias toward either end.
        distance = (config.NEAR_PLAYER_DISTANCE_M + config.FAR_PLAYER_DISTANCE_M) / 2
    return t_audio - distance / config.SPEED_OF_SOUND_MS


def segment(
    session_id: str,
    rally_break_s: float = config.RALLY_BREAK_S,
    use_motion: bool = False,
) -> PointsFile:
    paths = SessionPaths(session_id)
    hits_file = load_hits(session_id)
    events = sorted(hits_file.own_court, key=lambda h: h.t_audio)

    tr: motion.MotionTrace | None = None
    if use_motion:
        cache = paths.root / "motion.npz"
        if cache.exists():
            tr = motion.load(cache)
        else:
            tr = motion.trace(session_id)
            motion.save(tr, cache)

    points: list[Point] = []
    for p_index, (group, had_fault) in enumerate(_merge_faults(_cluster(events, rally_break_s))):
        shots: list[Shot] = []
        for s_index, hit in enumerate(group):
            player, confidence = (
                _attribute(hit, tr) if tr is not None else (Player.UNKNOWN, 0.0)
            )
            t_contact = _delay_corrected(hit.t_audio, player)
            shots.append(
                Shot(
                    index_in_point=s_index,
                    t_contact=t_contact,
                    t_start=t_contact - config.SHOT_PRE_S,
                    t_end=t_contact + config.SHOT_POST_S,
                    player=player,
                    # In match play every point opens with a serve, so the first
                    # event after a long gap is one by construction - no pose
                    # estimation needed for a first pass.
                    is_serve=(s_index == 0),
                    is_second_serve=(s_index == 0 and had_fault),
                    detect_confidence=confidence,
                )
            )

        points.append(
            Point(
                index=p_index,
                t_start=group[0].t_audio,
                t_end=group[-1].t_audio,
                server=shots[0].player if shots else Player.UNKNOWN,
                had_fault=had_fault,
                shots=shots,
            )
        )

    out = PointsFile(session_id=session_id, points=points)
    paths.points.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


def segment_from_serves(
    session_id: str,
    serve_times: list[float],
    tolerance_s: float = 0.30,
) -> PointsFile:
    """Re-derive point boundaries anchored on known serve times.

    Gap clustering cannot do this job. Measured over 26 minutes, the inter-event
    gap distribution is unimodal with a long tail - 131 gaps exceed 2.5s and 136
    more sit in the ambiguous 1.5-2.5s band, with no valley between them. Any
    threshold either splits rallies or merges points, and in practice it did
    both. The likely cause is the pre-serve routine: bouncing the ball before
    serving produces detected events that fill the silence a point boundary
    would otherwise show.

    In match play every point opens with a serve, so once labelling identifies
    serves this becomes exact. Call it with the serve timestamps from stage 5.
    """
    paths = SessionPaths(session_id)
    events = sorted(load_hits(session_id).own_court, key=lambda h: h.t_audio)
    serves = sorted(serve_times)

    if not serves:
        raise ValueError("No serve times supplied; nothing to anchor points on.")

    # Assign every event to the most recent serve at or before it. Events before
    # the first serve are warm-up and are dropped.
    groups: list[list[Hit]] = [[] for _ in serves]
    for hit in events:
        idx = int(np.searchsorted(serves, hit.t_audio + tolerance_s, side="right")) - 1
        if idx >= 0:
            groups[idx].append(hit)

    points: list[Point] = []
    pending_fault = False
    for serve_t, group in zip(serves, groups):
        if not group:
            continue
        # A serve whose "point" contains only the serve itself is a fault; its
        # events belong to the next point.
        is_fault = len(group) <= MAX_FAULT_EVENTS and group is not groups[-1]

        shots = [
            Shot(
                index_in_point=i,
                t_contact=_delay_corrected(h.t_audio, Player.UNKNOWN),
                t_start=_delay_corrected(h.t_audio, Player.UNKNOWN) - config.SHOT_PRE_S,
                t_end=_delay_corrected(h.t_audio, Player.UNKNOWN) + config.SHOT_POST_S,
                player=Player.UNKNOWN,
                is_serve=(i == 0),
                is_second_serve=(i == 0 and pending_fault),
            )
            for i, h in enumerate(group)
        ]
        points.append(
            Point(
                index=len(points),
                t_start=group[0].t_audio,
                t_end=group[-1].t_audio,
                had_fault=pending_fault,
                shots=shots,
            )
        )
        pending_fault = is_fault

    out = PointsFile(session_id=session_id, points=points)
    paths.points.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


def load(session_id: str) -> PointsFile:
    paths = SessionPaths(session_id)
    if not paths.points.exists():
        raise FileNotFoundError(
            f"No points.json for {session_id!r}. Run `tennis segment` first."
        )
    return PointsFile.model_validate_json(paths.points.read_text(encoding="utf-8"))


def summarise(pf: PointsFile) -> dict:
    shots = [s for p in pf.points for s in p.shots]
    lengths = [len(p.shots) for p in pf.points]
    by_player = {p: 0 for p in Player}
    for s in shots:
        by_player[s.player] += 1
    return {
        "points": len(pf.points),
        "shots": len(shots),
        "median_rally_length": float(np.median(lengths)) if lengths else 0.0,
        "longest_rally": max(lengths) if lengths else 0,
        "faults": sum(1 for p in pf.points if p.had_fault),
        "near": by_player[Player.NEAR],
        "far": by_player[Player.FAR],
        "unknown": by_player[Player.UNKNOWN],
    }
