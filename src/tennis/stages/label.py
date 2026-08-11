"""Stage 5 - label each contact sheet with a vision model.

This stage answers three questions that the local pipeline could not:

  1. **Is this a shot at all?** Detection is audio-based and deliberately
     high-recall, so the stream contains ball bounces, net cords and noise
     alongside real strikes. Acoustic features do not separate them.
  2. **Which player hit it?** Loudness only distinguishes our court from the
     neighbouring ones, and court-half motion energy was measured and found to
     be confidently wrong.
  3. **Which stroke was it?** The original goal.

Doing all three in one request is what makes the approach cheap: the image is
already being sent and dominates the token count, so the extra questions are
close to free. At roughly 1,150 image tokens per sheet, the whole corpus costs
about $3 through the Batches API.

Requests go through the Batches API rather than one at a time: this is not
latency-sensitive, and batching halves the price.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..config import SessionPaths

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 400

# The Batches API caps a request set at 256MB. Base64 inflates a ~160KB sheet to
# ~215KB, so the full corpus would exceed that in a single batch.
MAX_REQUESTS_PER_BATCH = 500
MAX_BATCH_BYTES = 180 * 1024 * 1024

SYSTEM = """\
You are analysing frames from a tennis video to identify what happened at a \
specific moment.

The camera is fixed behind the baseline. The player nearest the camera (large in \
frame, seen from behind) is the NEAR player. Their opponent, across the net and \
much smaller, is the FAR player.

Each image is a 3x3 grid of nine frames in chronological order, reading left to \
right, top to bottom. Each is labelled with its time offset from a candidate \
impact detected in the audio; the frame at that instant is labelled CONTACT and \
outlined in yellow. The crop follows the NEAR player, so the FAR player may be \
partly or wholly out of view.

The candidate was detected from sound alone, so it is often NOT a racket strike. \
It may be the ball bouncing on the court, the ball clipping the net, or nothing \
identifiable. Judge only what you can see.

Guidance:
- A strike shows a racket swinging through the ball: a backswing before contact \
and a follow-through after. A player standing, walking or waiting is not striking.
- If the NEAR player is clearly not swinging but the ball is visibly in play, the \
FAR player most likely hit it - report player "far" with event_type "strike".
- Judge the stroke from the swing across the whole sequence, not one frame. \
Remember the near player is seen FROM BEHIND: for a right-hander, a ball struck \
on the right side of their body is a forehand, on the left side a backhand.
- Set confidence "low" when the crop, motion blur or occlusion leave you unsure. \
An honest "low" is far more useful than a confident guess.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["strike", "bounce", "net_cord", "nothing"],
            "description": "What produced the detected sound.",
        },
        "player": {
            "type": "string",
            "enum": ["near", "far", "none"],
            "description": "Who struck the ball; 'none' if this was not a strike.",
        },
        "stroke": {
            "type": "string",
            "enum": [
                "forehand", "backhand", "serve", "volley",
                "overhead", "smash", "slice", "none",
            ],
            "description": "Stroke played; 'none' if not a strike or not visible.",
        },
        "quality": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "Technique and execution, 1 poor to 5 excellent. Use 3 if unsure.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {
            "type": "string",
            "description": "One short sentence on what is visible. Empty string if nothing to add.",
        },
    },
    "required": ["event_type", "player", "stroke", "quality", "confidence", "notes"],
    "additionalProperties": False,
}


@dataclass
class Labelled:
    custom_id: str
    sheet: Path
    label: dict


def _client():
    import anthropic

    return anthropic.Anthropic()


def _image_block(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def _params(sheet: Path, shot_index: int, is_serve: bool) -> dict:
    hint = (
        "This is the first event of a point, so it is likely a serve - but say so "
        "only if the swing actually looks like one."
        if is_serve
        else f"This is event {shot_index + 1} within the point."
    )
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [
            {
                "role": "user",
                "content": [
                    _image_block(sheet),
                    {"type": "text", "text": hint},
                ],
            }
        ],
    }


@dataclass
class Item:
    custom_id: str
    sheet: Path
    point_index: int
    shot_index: int
    is_serve: bool


def _sheet_index(session_id: str) -> list[Item]:
    """Pair every contact sheet with the shot it came from.

    `custom_id` is how batch results are matched back: they return in arbitrary
    order and must be keyed, never zipped. The API constrains it to
    ^[a-zA-Z0-9_-]{1,64}$, so it cannot carry the timestamp in the sheet's
    filename - point and shot index already identify a shot uniquely.
    """
    from .segment import load as load_points

    paths = SessionPaths(session_id)
    by_name = {p.stem: p for p in sorted(paths.sheets.glob("*.jpg"))}

    work: list[Item] = []
    for point in load_points(session_id).points:
        for shot in point.shots:
            stem = f"p{point.index:03d}_s{shot.index_in_point:02d}_t{shot.t_contact:08.2f}"
            sheet = by_name.get(stem)
            if sheet is not None:
                work.append(
                    Item(
                        custom_id=f"p{point.index:03d}_s{shot.index_in_point:02d}",
                        sheet=sheet,
                        point_index=point.index,
                        shot_index=shot.index_in_point,
                        is_serve=shot.is_serve,
                    )
                )
    return work


# --- immediate mode, for iterating on the prompt ---------------------------


def label_sync(session_id: str, limit: int = 8) -> list[Labelled]:
    """Label a handful of sheets immediately. Use this to check the prompt
    before committing to a batch that takes up to an hour."""
    client = _client()
    out: list[Labelled] = []
    for item in _sheet_index(session_id)[:limit]:
        response = client.messages.create(
            **_params(item.sheet, item.shot_index, item.is_serve)
        )
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        out.append(Labelled(item.custom_id, item.sheet, json.loads(text)))
    return out


# --- batch mode ------------------------------------------------------------


def _batch_state_path(session_id: str) -> Path:
    return SessionPaths(session_id).root / "label_batches.json"


def submit(session_id: str, limit: int | None = None) -> list[str]:
    """Submit every contact sheet, split across as many batches as needed."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    work = _sheet_index(session_id)
    if limit:
        work = work[:limit]
    if not work:
        raise RuntimeError("No contact sheets found. Run `tennis sheets` first.")

    chunks: list[list] = [[]]
    size = 0
    for item in work:
        params = _params(item.sheet, item.shot_index, item.is_serve)
        approx = len(params["messages"][0]["content"][0]["source"]["data"])
        if chunks[-1] and (
            len(chunks[-1]) >= MAX_REQUESTS_PER_BATCH or size + approx > MAX_BATCH_BYTES
        ):
            chunks.append([])
            size = 0
        chunks[-1].append(
            Request(
                custom_id=item.custom_id,
                params=MessageCreateParamsNonStreaming(**params),
            )
        )
        size += approx

    batch_ids = [client.messages.batches.create(requests=c).id for c in chunks if c]
    _batch_state_path(session_id).write_text(
        json.dumps({"batch_ids": batch_ids, "n_requests": len(work)}, indent=2),
        encoding="utf-8",
    )
    return batch_ids


def status(session_id: str) -> list[dict]:
    client = _client()
    state = json.loads(_batch_state_path(session_id).read_text(encoding="utf-8"))
    out = []
    for bid in state["batch_ids"]:
        b = client.messages.batches.retrieve(bid)
        counts = b.request_counts
        out.append(
            {
                "id": bid,
                "status": b.processing_status,
                "succeeded": counts.succeeded,
                "errored": counts.errored,
                "processing": counts.processing,
            }
        )
    return out


def collect(session_id: str) -> dict:
    """Gather finished results into labels.json, keyed by custom_id."""
    client = _client()
    paths = SessionPaths(session_id)
    state = json.loads(_batch_state_path(session_id).read_text(encoding="utf-8"))

    labels: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for bid in state["batch_ids"]:
        for result in client.messages.batches.results(bid):
            if result.result.type == "succeeded":
                msg = result.result.message
                text = next((b.text for b in msg.content if b.type == "text"), "{}")
                try:
                    labels[result.custom_id] = json.loads(text)
                except json.JSONDecodeError:
                    errors[result.custom_id] = f"unparseable: {text[:120]}"
            else:
                errors[result.custom_id] = result.result.type

    dest = paths.root / "labels.json"
    dest.write_text(
        json.dumps({"model": MODEL, "labels": labels, "errors": errors}, indent=2),
        encoding="utf-8",
    )
    return {"labelled": len(labels), "errors": len(errors), "path": str(dest)}
