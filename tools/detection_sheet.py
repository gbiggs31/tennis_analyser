"""Tile a video frame from every detected onset into one contact sheet.

Real footage has no ground truth, so the way to judge the detector is to look at
what it fired on. Each tile is labelled with its timestamp and index so a bad
detection can be traced straight back to hits.json.

Usage:
  uv run python tools/detection_sheet.py SESSION_ID [--cols 6] [--tile 480]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis.config import SessionPaths  # noqa: E402
from tennis.stages.hits import load as load_hits  # noqa: E402
from tennis.video import open_capture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=480, help="Tile width in pixels.")
    ap.add_argument("--crop", default=None,
                    help="Optional x0,y0,x1,y1 crop in proxy pixels before scaling.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--near-only", action="store_true",
                    help="Show only onsets on our court (above the level split).")
    ap.add_argument("--source", choices=("proxy", "original"), default="original",
                    help="The proxy is 30fps, so its nearest frame can be 16ms from "
                         "contact - far enough at 120fps to miss the strike entirely.")
    args = ap.parse_args()

    paths = SessionPaths(args.session)
    hits_file = load_hits(args.session)
    hits = hits_file.own_court if args.near_only else hits_file.hits
    if not hits:
        print("no hits")
        return 1

    if args.source == "original":
        probe = json.loads(paths.probe.read_text(encoding="utf-8"))
        video = Path(probe["path"])
        if not video.exists():
            print(f"original missing ({video}); falling back to proxy")
            video = paths.proxy
    else:
        video = paths.proxy

    # Proxies are already baked into final orientation; originals must be read
    # with the same rotation convention the ffmpeg pipeline uses.
    cap = open_capture(video, autorotate=(video == paths.proxy))

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None

    tiles: list[np.ndarray] = []
    for h in hits:
        cap.set(cv2.CAP_PROP_POS_MSEC, h.t_contact * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        if crop:
            x0, y0, x1, y1 = crop
            frame = frame[y0:y1, x0:x1]

        scale = args.tile / frame.shape[1]
        tile = cv2.resize(frame, (args.tile, int(frame.shape[0] * scale)))

        label = f"#{h.index}  {h.t_contact:.2f}s  {h.features.peak_db:.0f}dB"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()

    if not tiles:
        print("no frames read")
        return 1

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + tile.shape[0], c * tw : c * tw + tile.shape[1]] = tile

    out = Path(args.out) if args.out else paths.debug / "detections.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{out}  ({len(tiles)} tiles, {sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
