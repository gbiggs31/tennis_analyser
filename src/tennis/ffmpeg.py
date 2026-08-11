"""Thin wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from . import config

# winget drops the binaries under a versioned Links/Packages path that is added
# to PATH only for *new* shells, so look there too before giving up.
_WINGET_GLOBS = [
    Path.home() / "AppData/Local/Microsoft/WinGet/Links",
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
]


class FFmpegNotFound(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _resolve(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found

    exe = f"{name}.exe"
    for base in _WINGET_GLOBS:
        if not base.exists():
            continue
        direct = base / exe
        if direct.exists():
            return str(direct)
        for candidate in base.rglob(exe):
            return str(candidate)

    # Last resort: the imageio-ffmpeg wheel bundles a static ffmpeg (no ffprobe).
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    raise FFmpegNotFound(
        f"Could not find {name!r}. Install with:  winget install --id Gyan.FFmpeg "
        f"-e --source winget"
    )


def ffmpeg_path() -> str:
    return _resolve("ffmpeg")


def ffprobe_path() -> str:
    return _resolve("ffprobe")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}):\n{tail}")
    return proc


def _parse_rate(value: str | None) -> float | None:
    """ffprobe reports frame rates as 'num/den' strings."""
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe(video: Path) -> dict:
    """Return a normalised summary of the container's video and audio streams."""
    raw = json.loads(
        _run(
            [
                ffprobe_path(),
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video),
            ]
        ).stdout
    )

    streams = raw.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video}")

    fmt = raw.get("format", {})
    duration = float(fmt.get("duration") or video_stream.get("duration") or 0.0)

    # r_frame_rate is the container's nominal rate; avg_frame_rate is frames
    # divided by duration. They disagree on variable-frame-rate and on phone
    # slow-motion files, where the nominal rate is the *playback* rate.
    r_fps = _parse_rate(video_stream.get("r_frame_rate"))
    avg_fps = _parse_rate(video_stream.get("avg_frame_rate"))
    fps = avg_fps or r_fps or 0.0

    info = {
        "path": str(Path(video).resolve()),
        "size_bytes": int(fmt.get("size") or 0),
        "duration_s": duration,
        "fps": fps,
        "fps_nominal": r_fps,
        "fps_average": avg_fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "video_codec": video_stream.get("codec_name"),
        "n_frames": int(video_stream.get("nb_frames") or 0) or None,
        "rotation": _rotation(video_stream),
        "has_audio": audio_stream is not None,
    }

    if audio_stream is not None:
        info["audio"] = {
            "codec": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
            "duration_s": float(audio_stream.get("duration") or duration),
        }

    return info


def _rotation(stream: dict) -> int:
    """Phones record rotation as side data rather than swapping width/height."""
    for entry in stream.get("side_data_list") or []:
        if "rotation" in entry:
            try:
                return int(entry["rotation"]) % 360
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags") or {}
    try:
        return int(tags.get("rotate", 0)) % 360
    except (TypeError, ValueError):
        return 0


def extract_audio(video: Path, dest: Path, sample_rate: int = config.AUDIO_SAMPLE_RATE) -> Path:
    """Decode the audio track to mono PCM WAV at a known sample rate."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_path(), "-y",
            "-i", str(video),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(dest),
        ]
    )
    return dest


def _rotation_flags(autorotate: bool) -> list[str]:
    """Input-side flags controlling how the container's display matrix is treated.

    ffmpeg honours that matrix by default, and phone footage often carries a
    rotation that is wrong for our purposes. `-noautorotate` alone is not enough:
    it stops the *frames* being rotated but the side data is still copied to the
    output, so players rotate the result anyway. Overriding the declared rotation
    to zero suppresses both. Requires ffmpeg >= 6.0.
    """
    return [] if autorotate else ["-display_rotation", "0"]


def build_proxy(
    video: Path,
    dest: Path,
    width: int = config.PROXY_WIDTH,
    height: int = config.PROXY_HEIGHT,
    fps: int = config.PROXY_FPS,
    autorotate: bool = config.APPLY_ROTATION_DEFAULT,
) -> Path:
    """Downscaled, constant-frame-rate copy for CPU computer-vision work.

    Letterboxed rather than stretched so pixel coordinates map back to the
    original by a single uniform scale factor.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={fps}"
    )
    _run(
        [
            ffmpeg_path(), "-y",
            *_rotation_flags(autorotate),
            "-i", str(video),
            "-an",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(dest),
        ]
    )
    return dest


def cut_clip(
    video: Path,
    dest: Path,
    start_s: float,
    duration_s: float,
    autorotate: bool = config.APPLY_ROTATION_DEFAULT,
    mark_at_s: float | None = None,
    slowmo: float = 1.0,
) -> Path:
    """Frame-accurate cut from the original, re-encoded to keep every frame.

    Seeks input-side only. Since ffmpeg 2.1 that is both fast (keyframe jump,
    no decoding from zero) and accurate (decodes forward to the exact point) —
    the old advice to seek again after -i is obsolete, and actively harmful
    here: an output-side -ss leaves the filter graph's clock running from the
    *input* seek point, so a timed overlay lands in the wrong place.

    `mark_at_s` (relative to the clip) flashes a red border at the moment of
    contact, so it is immediately obvious whether the detection landed on the
    real impact. `slowmo` of 4 plays 120fps footage at quarter speed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, start_s)

    filters: list[str] = []
    if mark_at_s is not None:
        # Applied before setpts, so this is in source-time seconds from clip start.
        lo, hi = mark_at_s - 0.06, mark_at_s + 0.06
        filters.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=14:"
            f"enable='between(t,{lo:.3f},{hi:.3f})'"
        )
    if slowmo != 1.0:
        filters.append(f"setpts={slowmo:.4f}*PTS")

    cmd = [
        ffmpeg_path(), "-y",
        *_rotation_flags(autorotate),
        "-ss", f"{start_s:.4f}",
        "-i", str(video),
        "-t", f"{duration_s:.4f}",
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
    ]
    # setpts rescales video only; the audio would desync, so drop it in slow motion.
    cmd += ["-an"] if slowmo != 1.0 else ["-c:a", "aac"]
    cmd.append(str(dest))

    _run(cmd)
    return dest


def extract_frame(
    video: Path,
    dest: Path,
    t: float,
    scale_width: int | None = None,
    autorotate: bool = config.APPLY_ROTATION_DEFAULT,
) -> Path:
    """Grab a single frame at time `t`. Used for debug views and contact sheets."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(), "-y",
        *_rotation_flags(autorotate),
        "-ss", f"{max(0.0, t):.4f}",
        "-i", str(video),
        "-frames:v", "1",
    ]
    if scale_width:
        cmd += ["-vf", f"scale={scale_width}:-2"]
    cmd.append(str(dest))
    _run(cmd)
    return dest
