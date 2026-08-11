"""Generate a synthetic tennis clip with known ground-truth hit times.

Real footage is the eventual test, but a fixture with labelled impacts lets us
measure detector precision and recall numerically, and catch regressions when
thresholds get retuned.

The simulation reproduces the acoustic structure the detector relies on:

  * racket strikes are bright, fast-attack transients
  * court bounces are duller and quieter, and fall between strikes
  * the far player is ~16dB down and ~64ms late (distance and speed of sound)
  * points are separated by long quiet gaps, each opening with a serve

Usage:  uv run python tools/make_fixture.py [--out DIR] [--noise 0.01]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis import config, ffmpeg  # noqa: E402

SR = config.AUDIO_SAMPLE_RATE
FPS = 120
WIDTH, HEIGHT = 640, 360

RNG = np.random.default_rng(20260811)


def _impact(kind: str, sr: int = SR) -> np.ndarray:
    """A filtered, exponentially-decaying noise burst.

    That is physically what these sounds are: a broadband click shaped by the
    stiffness of whatever was struck.
    """
    if kind == "racket":
        duration, tau, band = 0.045, 0.007, (900.0, 9000.0)
    elif kind == "bounce":
        duration, tau, band = 0.060, 0.016, (250.0, 2600.0)
    else:
        raise ValueError(kind)

    n = int(duration * sr)
    t = np.arange(n) / sr
    noise = RNG.standard_normal(n)

    nyq = sr / 2
    sos = signal.butter(4, [band[0] / nyq, min(band[1], nyq * 0.99) / nyq],
                        btype="band", output="sos")
    shaped = signal.sosfilt(sos, noise)

    # ~1ms attack ramp then exponential decay
    attack = int(0.001 * sr)
    env = np.exp(-t / tau)
    env[:attack] *= np.linspace(0.0, 1.0, attack)

    out = shaped * env
    peak = np.max(np.abs(out))
    return (out / peak).astype(np.float32) if peak > 0 else out.astype(np.float32)


def _add(buf: np.ndarray, sound: np.ndarray, t: float, gain: float) -> None:
    start = int(round(t * SR))
    end = min(start + len(sound), len(buf))
    if start < len(buf) and end > start:
        buf[start:end] += sound[: end - start] * gain


def build_audio(noise_level: float) -> tuple[np.ndarray, list[dict]]:
    """Lay out several points and return the audio plus ground-truth events."""
    racket = _impact("racket")
    bounce = _impact("bounce")

    # Attenuation and propagation delay follow from the court geometry in config.
    near_d = config.NEAR_PLAYER_DISTANCE_M
    far_d = config.FAR_PLAYER_DISTANCE_M
    far_gain = (near_d / far_d) ** 2          # inverse square, ~ -16 dB
    near_delay = near_d / config.SPEED_OF_SOUND_MS
    far_delay = far_d / config.SPEED_OF_SOUND_MS

    rally_lengths = [4, 6, 3, 7, 2]           # shots per point
    faults = {1}                              # point index that opens with a fault
    events: list[dict] = []

    t = 3.0
    for p_index, n_shots in enumerate(rally_lengths):
        server = "near" if p_index % 2 == 0 else "far"

        if p_index in faults:
            # A fault: serve, then a short pause, then the second serve.
            events.append({"t_contact": t, "player": server, "kind": "racket",
                           "is_serve": True, "point": p_index})
            t += 2.2

        player = server
        for s_index in range(n_shots):
            events.append({
                "t_contact": t,
                "player": player,
                "kind": "racket",
                "is_serve": s_index == 0,
                "point": p_index,
            })
            # The ball bounces roughly midway before reaching the opponent.
            if s_index < n_shots - 1:
                events.append({"t_contact": t + 0.42, "player": None, "kind": "bounce",
                               "point": p_index})
            t += float(RNG.uniform(0.80, 1.10))
            player = "far" if player == "near" else "near"

        t += float(RNG.uniform(7.0, 9.5))     # between-point pause

    total = t + 3.0
    buf = np.zeros(int(total * SR), dtype=np.float32)

    # Broadband ambience plus low-frequency wind rumble.
    buf += RNG.standard_normal(len(buf)).astype(np.float32) * noise_level
    nyq = SR / 2
    rumble = signal.sosfilt(
        signal.butter(2, 200.0 / nyq, btype="low", output="sos"),
        RNG.standard_normal(len(buf)),
    ).astype(np.float32)
    buf += rumble * noise_level * 6.0

    for ev in events:
        if ev["kind"] == "racket":
            gain = 0.85 if ev["player"] == "near" else 0.85 * far_gain
            delay = near_delay if ev["player"] == "near" else far_delay
            sound = racket
        else:
            # Bounces happen mid-court; treat them as a fixed mid distance.
            gain = 0.30 * ((near_d / 15.0) ** 2)
            delay = 15.0 / config.SPEED_OF_SOUND_MS
            sound = bounce
        gain *= float(RNG.uniform(0.80, 1.20))
        ev["t_audio"] = ev["t_contact"] + delay
        _add(buf, sound, ev["t_audio"], gain)

    peak = float(np.max(np.abs(buf)))
    if peak > 0.95:
        buf *= 0.95 / peak
    return buf, events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=config.RAW_DIR)
    parser.add_argument("--noise", type=float, default=0.010,
                        help="Broadband noise floor amplitude (try 0.05 for a windy day).")
    parser.add_argument("--name", default="fixture_rally")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, events = build_audio(args.noise)
    duration = len(audio) / SR

    import soundfile as sf

    wav = out_dir / f"{args.name}.wav"
    sf.write(str(wav), audio, SR, subtype="PCM_16")

    video = out_dir / f"{args.name}.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(), "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={duration:.3f}",
            "-i", str(wav),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    wav.unlink()

    truth = out_dir / f"{args.name}.truth.json"
    strikes = [e for e in events if e["kind"] == "racket"]
    truth.write_text(
        json.dumps(
            {
                "video": str(video),
                "duration_s": duration,
                "noise_level": args.noise,
                "n_strikes": len(strikes),
                "n_bounces": len(events) - len(strikes),
                "events": events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"video   {video}")
    print(f"truth   {truth}")
    print(f"        {duration:.1f}s, {len(strikes)} strikes, "
          f"{len(events) - len(strikes)} bounces, noise={args.noise}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
