# tennis_analyser

Turns long tennis videos into a folder of **labelled clips of your shots**.

```sh
uv run tennis ingest  "path/to/match.mp4"   # audio + analysis proxy
uv run tennis hits    <session>             # find ball contacts in the audio
uv run tennis segment <session>             # group them into points
uv run tennis sheets  <session>             # one contact sheet per shot
uv run tennis label   <session> test        # check the prompt on a few (cheap)
uv run tennis label   <session> submit      # then the whole session, batched
uv run tennis label   <session> collect
uv run tennis shots   <session> --dry-run   # what would be exported
uv run tennis shots   <session>             # clips, one folder per stroke
```

Everything except `label` runs locally and free.

Shot from a fixed camera behind the baseline, a 1080p/120fps match recording is far too
large to hand to a vision model directly. This pipeline segments it locally, then sends
only tiny per-shot summaries to an LLM for labelling.

## How it works

The spine is **audio**. A racket-ball impact is a sharp broadband transient, and onset
detection on the audio track finds those impacts in seconds on a CPU — no GPU, no ball
tracking model. Pose, motion and vision are layered on top as refinement.

```
0  ingest    video.mov ──▶ audio.wav + proxy.mp4 + probe.json
1  hits      audio.wav ──▶ hits.json      candidate impacts + acoustic features
2  segment   hits.json ──▶ points.json    serves detected, hits grouped into points
3  (review UI - not built; validation was done with contact sheets instead)
4  extract   clips/*.mp4 + sheets/*.jpg   per shot
5  label     contact sheets ──▶ stroke labels via the Claude Batch API
6  query     "top 5 forehands", coaching export
```

Two design notes worth knowing before reading the code:

- **Time is always seconds (float)**, never frame indices. The pipeline mixes a 120fps
  original with a 30fps proxy; frame numbers across those timebases invite off-by-N bugs.
- **Each stage writes an artifact to disk** and reads only the previous stage's output.
  Thresholds get re-tuned many times; re-running stage 2 must never re-decode video.

Player attribution was expected to come free from the camera position — the far player
is ~26 m away against ~4 m for the near player, so their strikes should arrive quieter
and later. **It does not work.** Loudness separates our court from the neighbouring ones
and nothing finer, and a motion-based fallback scored zero out of five against
hand-verified shots. Attribution is done by the vision model instead; see below.

## What the footage told us

Measured on `VID20260810202344~2` (24s, 1080p120, evening club session):

| Finding | Consequence |
|---|---|
| Audio is clean — 20% of total energy sits in the 700Hz-10kHz impact band | Audio-first detection is viable; no GPU or ball tracker needed |
| Onset timing is accurate to ~5ms against a synthetic fixture | The extracted contact frame is the real one, even at 120fps |
| Peak level splits cleanly into two populations, ~14dB apart | An automatic Otsu split on `peak_db` isolates near-court events |
| Adjacent courts are audible | They land in the quiet population, below the split, and are rejected |
| The camera is fixed (background varies <2/255 over 24s) | A one-off court calibration and background subtraction are both viable |
| The phone tags every clip `rotation=270`, which is **wrong** | The pipeline overrides it; see the orientation note below |

### What the level split does *not* do

Reviewing real clips showed the loud population mixes near-side shots, far-side
shots, ball bounces and net cords. Distance to the microphone separates our court
from the courts beside it, and nothing finer.

Two attempts to go finer failed, and both are worth recording so they are not
retried:

- **Acoustic features.** Attack time, spectral centroid and HF/LF ratio overlap
  almost completely between strikes and bounces on real footage.
- **Court-half motion energy.** Measured against five hand-verified near-player
  strikes, it got all five backwards. Each half must be normalised against its
  own baseline to be comparable, but the near player's constant movement inflates
  his own baseline until a swing barely registers (dynamic range 1.00→1.59),
  while the near-static far half spikes at anything that moves, including the
  adjacent court (range 1.00→7.54). The normalisation destroys the signal.

So **stage 2 does grouping only**, and both event type and player attribution move
to the labelling stage. A vision model reading the contact sheet can see which
player is swinging and whether it was a strike at all — and it is already being
asked which stroke it was, so this costs nothing extra. At ~1,900 input tokens
per event, labelling all 1,538 events in the current corpus is about $4, or $2
batched. That makes high recall the right trade: over-detect, and let labelling
sort it out.

### Why point boundaries need serves, not gaps

Stage 2 groups events by silence, and that is known to be imperfect. Measured
over the whole corpus, the inter-event gap distribution is **unimodal with a long
tail**: 131 gaps exceed 2.5s and 136 more sit in the ambiguous 1.5-2.5s band,
with no valley between them. Confirmed in review — a 52-second "point" turned out
to be two points whose boundary never opened a gap wider than 2.49s.

The likely cause is the pre-serve routine: bouncing the ball before serving
produces detected events that fill the silence a boundary would otherwise show.

In match play every point opens with a serve, so `segment_from_serves()` re-derives
boundaries exactly once labelling has identified them. Gap clustering is the
provisional pass, not the answer.

### Orientation

Two independent traps, both handled centrally:

- ffmpeg honours the container's rotation flag, and `-noautorotate` alone is not
  enough — it stops the frames rotating but still copies the flag to the output,
  so players rotate the result anyway. We override with `-display_rotation 0`.
- OpenCV *also* auto-rotates, independently. Every capture must go through
  `tennis.video.open_capture`, which matches the ffmpeg convention. Otherwise the
  same video is upright on one path and sideways on the other, and pixel
  coordinates stop agreeing between the proxy and the original.

## Setup

Requires Python 3.12 (MediaPipe has no 3.13+ wheels) and ffmpeg.

```sh
winget install --id Gyan.FFmpeg -e --source winget
uv sync
uv run tennis doctor
```

Raw video and derived artifacts live **outside** the repo, under `C:\tennis_data` by
default — a 30-minute 120fps clip is 8-15 GB and this repo sits in a synced folder.
Override with the `TENNIS_DATA_ROOT` environment variable.

## Usage

```sh
uv run tennis ingest "C:\tennis_data\raw\match.mov"
```

`ingest` reports whether the audio track is actually usable for hit detection — impact-band
energy, clipping, silence — which determines whether the rest of the pipeline takes the
cheap path or falls back to motion analysis.

```sh
uv run tennis sessions                      # what has been ingested
uv run tennis hits <session-id> -k 6        # stage 1: detect onsets
uv run tennis peek <session-id> --at 8      # sanity-check framing
```

### Tuning and validation

There is no ground truth for real footage, so two tools stand in:

```sh
# Synthetic clip with known hit times, for regression testing and threshold sweeps
uv run python tools/make_fixture.py
uv run tennis hits <fixture-session> --truth C:\tennis_data\raw\fixture_rally.truth.json \
    --sweep 1:8:0.5

# Every detection as a labelled video frame - the fastest way to judge real footage
uv run python tools/detection_sheet.py <session-id> --near-only
```

The fixture is a sanity check, not a target: its noise is synthetic and tuning hard
against it would overfit. Use it to catch regressions, and the detection sheet
(and later the review UI) to judge real recordings.

## Results on 26 minutes of footage

| | |
|---|---|
| Audio onsets detected | 3,412 |
| On our court (level split) | 1,538 |
| Labelled by the vision model | 1,538, zero errors |
| Identified as the near player's shots | 283 |
| Clips exported after de-duplication | 225 |

Stroke distribution after the second pass: 74 forehand, 69 backhand, 40 serve,
11 volley, 1 overhead, with 30 quarantined in `review/`.

**The stroke labels are much better than the first pass, but are not verified
truth.** The first pass produced 212 forehands to 10 backhands - a ratio no club
player generates - and re-running the strikes through a stronger model changed
50-55% of them, giving the near-even split above. That shift is good evidence
the correction worked. Individual labels are another matter: spot-checking six
contact frames by eye, only two could be called confidently either way, because
a forehand's follow-through crosses to the left shoulder and looks like a
backhand in a still. Treat `review/` as needing eyes, and expect some errors
outside it.

What is reliable: the clips are real shots, cut at the right moment, at full
120fps with lead-in and follow-through.
