"""Stage 4 - build one contact sheet per event, for the labelling model.

A shot is only legible as a sequence: backswing, contact, follow-through. A
single frame at the instant of impact cannot distinguish a forehand from a
backhand, nor a strike from a player standing still while the ball bounces.

Sending video is out of the question at this volume, so each event becomes a
single tiled image of frames spanning the swing. That is ~1,900 input tokens,
against ~$0.0013 batched - cheap enough to label every detected event and let
the model discard the ones that are not shots.

Frames are sampled unevenly, densest around contact, because that is where the
information is: the racket moves further in the 50ms around impact than in the
half-second of preparation before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config, players
from ..config import SessionPaths
from ..models import Player, Shot
from ..video import capture, load_probe
from .segment import load as load_points

# Offsets from contact, in seconds. Dense near zero, sparse in the wind-up.
FRAME_OFFSETS = (-0.50, -0.33, -0.20, -0.10, -0.04, 0.0, 0.05, 0.13, 0.28)

# Portrait tiles: a player's bounding box is roughly 1:3, so landscape tiles
# letterbox away most of their pixels - and pixels are tokens.
TILE_W, TILE_H = 260, 364
GRID_COLS, GRID_ROWS = 3, 3
LABEL_H = 18

# Widen the crop more than it is heightened, so the tile keeps some court either
# side of the player (where the ball is) without adding empty sky above them.
CROP_ASPECT = TILE_W / TILE_H


@dataclass
class SheetResult:
    point_index: int
    shot: Shot
    path: Path
    player_located: bool


def _crop_for(
    frame: np.ndarray,
    box: players.Box | None,
    scale_to_original: float,
    factor: float = config.SHEET_BBOX_SCALE,
) -> np.ndarray:
    """Crop the original-resolution frame around a proxy-space box."""
    H, W = frame.shape[:2]
    if box is None:
        # No player located: fall back to the court region so the tile still
        # shows whether a ball is in play.
        return frame[int(H * 0.35) :, :]
    scaled = box.to_scale(scale_to_original).scaled(factor, (W, H))

    # Pad out to the tile's aspect ratio before cropping, so the extra pixels
    # become court context rather than letterbox bars.
    if scaled.h > 0 and scaled.w / scaled.h < CROP_ASPECT:
        new_w = min(scaled.h * CROP_ASPECT, W)
        x = min(max(0.0, scaled.cx - new_w / 2), W - new_w)
        scaled = players.Box(int(x), scaled.y, int(new_w), scaled.h)

    if scaled.w < 16 or scaled.h < 16:
        return frame[int(H * 0.35) :, :]
    return frame[scaled.y : scaled.y + scaled.h, scaled.x : scaled.x + scaled.w]


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox into a fixed tile so the grid stays aligned."""
    if img.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(
        img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
    )
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - resized.shape[0]) // 2
    x = (w - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


def build(
    session_id: str,
    which: Player = Player.NEAR,
    limit: int | None = None,
    quality: int = 82,
) -> list[SheetResult]:
    """Render a contact sheet for every shot in points.json."""
    paths = SessionPaths(session_id)
    probe = load_probe(session_id)
    source = Path(probe.path)
    if not source.exists():
        raise FileNotFoundError(f"Source video missing: {source}")

    background = players.load_or_build_background(session_id)
    proxy_h, proxy_w = background.shape
    scale_to_original = probe.width / proxy_w

    work = [(p.index, s) for p in load_points(session_id).points for s in p.shots]
    if limit:
        work = work[:limit]

    out_dir = paths.sheets
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[SheetResult] = []

    # One capture for the original and one for the proxy: the proxy locates the
    # player cheaply, the original supplies the pixels.
    with capture(source, autorotate=probe.autorotate_applied) as cap_src, \
            capture(paths.proxy, autorotate=True) as cap_proxy:

        for point_index, shot in work:
            tiles: list[np.ndarray] = []
            located = False

            # Locate the player once, at contact, and reuse that crop for every
            # frame. Re-finding per frame would make the crop jitter, which
            # reads as camera shake and obscures the swing.
            cap_proxy.set(cv2.CAP_PROP_POS_MSEC, shot.t_contact * 1000.0)
            ok_p, proxy_frame = cap_proxy.read()
            box = None
            if ok_p:
                box = players.find(
                    cv2.cvtColor(proxy_frame, cv2.COLOR_BGR2GRAY), background, which
                )
                located = box is not None

            for offset in FRAME_OFFSETS:
                t = max(0.0, shot.t_contact + offset)
                cap_src.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok_s, frame = cap_src.read()
                if not ok_s:
                    tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
                    continue
                tile = _fit(_crop_for(frame, box, scale_to_original), TILE_W, TILE_H)
                _annotate(tile, offset)
                tiles.append(tile)

            sheet = _compose(tiles)
            name = (
                f"p{point_index:03d}_s{shot.index_in_point:02d}"
                f"_t{shot.t_contact:08.2f}.jpg"
            )
            dest = out_dir / name
            cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
            results.append(SheetResult(point_index, shot, dest, located))

    return results


def _annotate(tile: np.ndarray, offset: float) -> None:
    """Label each tile with its offset from contact, and ring the contact frame."""
    text = "CONTACT" if offset == 0.0 else f"{offset * 1000:+.0f}ms"
    cv2.rectangle(tile, (0, 0), (tile.shape[1], LABEL_H), (0, 0, 0), -1)
    colour = (0, 215, 255) if offset == 0.0 else (235, 235, 235)
    cv2.putText(tile, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    if offset == 0.0:
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 215, 255), 2)


def _compose(tiles: list[np.ndarray]) -> np.ndarray:
    rows = []
    for r in range(GRID_ROWS):
        row = tiles[r * GRID_COLS : (r + 1) * GRID_COLS]
        while len(row) < GRID_COLS:
            row.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)
