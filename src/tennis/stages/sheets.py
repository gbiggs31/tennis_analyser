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

**Tiles show the whole court, not a crop of one player.** Cropping was tried
first, on the reasoning that the far player is only ~30 proxy pixels tall and
would be illegible otherwise. It went badly. Locating a player needs background
subtraction, which needs a background that tracks the changing evening light,
and even once that was fixed the detector found a person-shaped blob in only 15%
of frames - so most sheets fell back to an uncropped view regardless.

Dropping the top third of the frame (sky, trees, houses) instead costs *fewer*
tokens than the cropped version, because it removes the letterbox padding a
portrait crop needed. It also cannot fail, shows both players so the model can
attribute the shot, and keeps the court visible so a ball bounce is
recognisable. The player-crop path is retained as an option but is not default.
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

# Six frames instead of nine for the strip layout: the swing is still legible,
# and the tokens freed go into making each frame bigger. Player size turned out
# to matter far more than frame count - a nine-frame wide view scored 0/12 on
# shots a six-frame close view gets right.
STRIP_OFFSETS = (-0.33, -0.15, -0.05, 0.0, 0.08, 0.22)

# Fixed region holding both players and no sky, as fractions of the frame.
# Fixed rather than detected: locating the player by background subtraction
# failed on long sessions (changing light) and then found a person in only 15%
# of frames once fixed. A constant crop cannot fail.
STRIP_BOX = (0.06, 0.52, 0.94, 1.00)

# Frames used to locate the player. Spread around contact so one bad frame
# (occlusion, a shadow) cannot decide the crop on its own.
LOCATE_OFFSETS = (-0.20, -0.04, 0.13)

GRID_COLS, GRID_ROWS = 3, 3
LABEL_H = 18

# Everything above this fraction of frame height is sky, trees, houses and the
# adjacent courts. Dropping it is free information-wise and makes both players
# proportionally larger in the tile.
COURT_TOP = 0.28

# Sized so the finished sheet is just under the 1568px long edge that images are
# downscaled to - going wider would be re-encoded away, not shown to the model.
TILE_W = 522
COURT_TILE_H = 211          # 1920x778 court crop, scaled to width
STRIP_COLS, STRIP_ROWS = 2, 3
STRIP_TILE_W, STRIP_TILE_H = 783, 258
PLAYER_TILE_W, PLAYER_TILE_H = 260, 364
CROP_ASPECT = PLAYER_TILE_W / PLAYER_TILE_H


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
    crop: str = "court",
) -> list[SheetResult]:
    """Render a contact sheet for every shot in points.json.

    `crop="court"` (default) keeps the whole court minus the sky: robust, shows
    both players, and cheaper than the alternative. `crop="player"` zooms on one
    player via background subtraction, which is tighter when it works but often
    does not - see the module docstring.
    """
    paths = SessionPaths(session_id)
    probe = load_probe(session_id)
    source = Path(probe.path)
    if not source.exists():
        raise FileNotFoundError(f"Source video missing: {source}")

    if crop == "strip":
        tile_w, tile_h = STRIP_TILE_W, STRIP_TILE_H
        offsets, cols, rows = STRIP_OFFSETS, STRIP_COLS, STRIP_ROWS
    elif crop == "court":
        tile_w, tile_h = TILE_W, COURT_TILE_H
        offsets, cols, rows = FRAME_OFFSETS, GRID_COLS, GRID_ROWS
    else:
        tile_w, tile_h = PLAYER_TILE_W, PLAYER_TILE_H
        offsets, cols, rows = FRAME_OFFSETS, GRID_COLS, GRID_ROWS

    background = None
    proxy_w = probe.proxy_width or config.PROXY_WIDTH
    proxy_h = probe.proxy_height or config.PROXY_HEIGHT
    if crop == "player":
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

            box = None
            if crop == "player":
                # Locate across a few frames and take the median box, then reuse
                # it for every tile. Re-finding per frame makes the crop jitter,
                # which reads as camera shake and obscures the swing.
                probe_frames: list[tuple[float, np.ndarray]] = []
                for offset in LOCATE_OFFSETS:
                    t = max(0.0, shot.t_contact + offset)
                    cap_proxy.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ok_p, proxy_frame = cap_proxy.read()
                    if ok_p:
                        probe_frames.append(
                            (t, cv2.cvtColor(proxy_frame, cv2.COLOR_BGR2GRAY))
                        )
                box, _ = players.find_stable(probe_frames, background, which)
                located = box is not None
                if box is None:
                    box = players.fallback_box(which, proxy_w, proxy_h)

            for offset in offsets:
                t = max(0.0, shot.t_contact + offset)
                cap_src.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok_s, frame = cap_src.read()
                if not ok_s:
                    tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                    continue
                H, W = frame.shape[:2]
                if crop == "strip":
                    x0, y0, x1, y1 = STRIP_BOX
                    region = frame[int(H * y0) : int(H * y1), int(W * x0) : int(W * x1)]
                elif crop == "court":
                    region = frame[int(H * COURT_TOP) :, :]
                else:
                    region = _crop_for(frame, box, scale_to_original)
                tile = _fit(region, tile_w, tile_h)
                _annotate(tile, offset)
                tiles.append(tile)

            sheet = _compose(tiles, cols, rows)
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


def _compose(tiles: list[np.ndarray], cols: int, rows: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    out = []
    for r in range(rows):
        row = tiles[r * cols : (r + 1) * cols]
        while len(row) < cols:
            row.append(np.zeros((h, w, 3), dtype=np.uint8))
        out.append(np.hstack(row))
    return np.vstack(out)
