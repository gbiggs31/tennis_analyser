"""Command line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import config, ffmpeg
from .models import Player

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Segment tennis footage into points and shots, label them, and query the result.",
)
console = Console()


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_duration(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


@app.command()
def doctor() -> None:
    """Check that external tools and data directories are ready."""
    table = Table(show_header=False, box=None)
    ok = True

    for name, getter in (("ffmpeg", ffmpeg.ffmpeg_path), ("ffprobe", ffmpeg.ffprobe_path)):
        try:
            table.add_row(f"[green]OK[/]  {name}", getter())
        except ffmpeg.FFmpegNotFound as exc:
            ok = False
            table.add_row(f"[red]MISSING[/]  {name}", str(exc))

    root = config.DATA_ROOT
    state = "exists" if root.exists() else "will be created on first ingest"
    table.add_row("[green]OK[/]  data root", f"{root}  ({state})")

    # Only the labelling stage needs credentials, so a missing key is a warning
    # rather than a failure. Never print the key itself - just enough to confirm
    # the right one was picked up.
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    env_file = config.REPO_ROOT / ".env"
    if key:
        source = ".env" if env_file.exists() else "shell environment"
        table.add_row(
            "[green]OK[/]  api key",
            f"{key[:11]}...{key[-4:]}  ({len(key)} chars, from {source})",
        )
    else:
        table.add_row(
            "[yellow]--[/]  api key",
            "not set - only needed for labelling. Put it in .env (not .env.example).",
        )

    # .env.example is tracked; .env is not. Putting a real key in the former is
    # an easy mistake and would publish it on the next push.
    example = config.REPO_ROOT / ".env.example"
    if example.exists() and "sk-ant-api" in example.read_text(encoding="utf-8"):
        table.add_row(
            "[bold red]LEAK[/]  .env.example",
            "contains what looks like a real key. That file is TRACKED BY GIT - "
            "move the key to .env, restore the template, and rotate the key.",
        )
        ok = False

    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    video: Annotated[Path, typer.Argument(help="Path to the source video file.")],
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild audio and proxy even if they exist.")
    ] = False,
    autorotate: Annotated[
        bool,
        typer.Option(
            "--autorotate/--no-autorotate",
            help="Honour the container's rotation flag. Off by default: the source "
            "phone tags clips with a rotation that turns the court on its side.",
        ),
    ] = config.APPLY_ROTATION_DEFAULT,
) -> None:
    """Stage 0: probe the video, extract audio, and build the analysis proxy."""
    from .stages.ingest import ingest as run_ingest

    with console.status(f"Ingesting {video.name} ..."):
        result = run_ingest(video, force=force, autorotate=autorotate)

    p = result.probe
    table = Table(title=f"Session {result.session_id}", show_header=False, box=None)
    table.add_row("source", p.path)
    table.add_row("size", _fmt_bytes(p.size_bytes))
    table.add_row("duration", f"{_fmt_duration(p.duration_s)}  ({p.duration_s:.3f}s)")
    table.add_row("resolution", f"{p.width}x{p.height}  {p.video_codec}")
    fps_detail = ""
    if p.fps_nominal and p.fps_average and abs(p.fps_nominal - p.fps_average) > 0.01:
        fps_detail = f"  (nominal {p.fps_nominal:.2f} / average {p.fps_average:.2f})"
    table.add_row("frame rate", f"{p.fps:.2f} fps{fps_detail}")
    console.print(table)
    console.print()

    a = result.audio
    if a is None:
        console.print("[bold red]audio[/]  none")
    else:
        colour = "red" if a.silent else ("yellow" if a.band_energy_fraction < 0.03 else "green")
        atable = Table(title="Audio", show_header=False, box=None)
        atable.add_row("codec", f"{p.audio.get('codec') if p.audio else '?'}")
        atable.add_row("sample rate", f"{a.sample_rate} Hz")
        atable.add_row("duration", f"{a.duration_s:.3f}s")
        atable.add_row("peak / rms", f"{a.peak:.4f}  /  {a.rms_db:.1f} dB")
        atable.add_row("impact-band energy", f"{a.band_energy_fraction * 100:.2f}%")
        if a.clipped_fraction > 0.001:
            atable.add_row("clipped samples", f"{a.clipped_fraction * 100:.2f}%")
        atable.add_row("verdict", f"[{colour}]{a.verdict}[/]")
        console.print(atable)

    console.print()
    console.print(f"artifacts  {result.paths.root}")
    for label, path in (("audio", result.paths.audio), ("proxy", result.paths.proxy)):
        if path.exists():
            console.print(f"  {label:<6} {path.name}  ({_fmt_bytes(path.stat().st_size)})")

    for warning in result.warnings:
        console.print(f"[yellow]warning[/]  {warning}")


@app.command()
def hits(
    session: Annotated[str, typer.Argument(help="Session id (see `tennis sessions`).")],
    k: Annotated[
        float,
        typer.Option("-k", "--threshold-k", help="MADs above the local noise floor."),
    ] = 6.0,
    truth: Annotated[
        Path | None,
        typer.Option("--truth", help="Ground-truth JSON to score against (fixtures)."),
    ] = None,
    beeps: Annotated[
        bool, typer.Option("--beeps", help="Also render an audio track with a tone per hit.")
    ] = False,
    plot_window: Annotated[
        str | None,
        typer.Option("--plot-window", help="Restrict the debug plot, e.g. '10:40'."),
    ] = None,
    sweep: Annotated[
        str | None,
        typer.Option("--sweep", help="Score a range of k against --truth, e.g. '1.5:8:0.5'."),
    ] = None,
) -> None:
    """Stage 1: detect candidate ball contacts in the audio."""
    from .stages import hits as stage

    if sweep:
        if not truth:
            console.print("[red]--sweep needs --truth to score against.[/]")
            raise typer.Exit(code=1)
        lo, hi, step = (float(v) for v in sweep.split(":"))
        ks = [round(lo + i * step, 3) for i in range(int((hi - lo) / step) + 1)]
        with console.status(f"Sweeping {len(ks)} thresholds ..."):
            rows = stage.sweep(session, truth, ks)

        table = Table(title="Threshold sweep")
        for col in ("k", "detected", "matched", "recall", "precision", "F1", "bounces", "other FP"):
            table.add_column(col, justify="right")
        best = max(rows, key=lambda r: r[1].f1)[0]
        for kv, ev in rows:
            mark = "[bold green]" if kv == best else ""
            end = "[/]" if mark else ""
            table.add_row(
                f"{mark}{kv:g}{end}",
                str(ev.n_detected), str(ev.matched),
                f"{mark}{ev.recall * 100:.1f}%{end}",
                f"{mark}{ev.precision * 100:.1f}%{end}",
                f"{mark}{ev.f1 * 100:.1f}%{end}",
                str(ev.bounces_detected), str(len(ev.spurious)),
            )
        console.print(table)
        return

    with console.status("Detecting onsets ..."):
        result = stage.detect(session, threshold_k=k)

    hf = result.hits_file
    n = len(hf.hits)
    own = len(hf.own_court)
    duration = hf.audio_duration_s
    console.print(
        f"{n} onsets over {duration:.1f}s  ({n / duration * 60:.1f}/min, k={k})"
    )
    if hf.own_court_threshold_db is not None:
        console.print(
            f"[bold]{own}[/] on our court "
            f"({own / duration * 60:.1f}/min)  "
            f"level split at {hf.own_court_threshold_db:.1f} dB"
        )

    window = None
    if plot_window:
        lo, _, hi = plot_window.partition(":")
        window = (float(lo), float(hi))
    paths = config.SessionPaths(session)
    plot = stage.plot_flux(result, paths.debug / "flux.png", truth_path=truth, window=window)
    console.print(f"plot   {plot}")

    if beeps:
        wav = stage.render_beeps(session, paths.debug / "beeps.wav", result.hits_file.hits)
        console.print(f"beeps  {wav}")

    if truth:
        ev = stage.evaluate(result.hits_file, truth)
        table = Table(title="Against ground truth", show_header=False, box=None)
        table.add_row("true strikes", str(ev.n_truth))
        table.add_row("detected", str(ev.n_detected))
        table.add_row("matched", str(ev.matched))
        table.add_row("recall", f"{ev.recall * 100:.1f}%")
        table.add_row("precision", f"{ev.precision * 100:.1f}%")
        table.add_row("F1", f"{ev.f1 * 100:.1f}%")
        table.add_row("median timing error", f"{ev.median_error_ms:.1f} ms")
        table.add_row("bounces picked up", str(ev.bounces_detected))
        table.add_row("other false positives", str(len(ev.spurious)))
        console.print()
        console.print(table)
        if ev.missed:
            preview = ", ".join(f"{t:.2f}s" for t in ev.missed[:10])
            console.print(f"[yellow]missed[/]  {preview}")


@app.command()
def segment(
    session: Annotated[str, typer.Argument(help="Session id (see `tennis sessions`).")],
    gap: Annotated[
        float, typer.Option("--gap", help="Silence (s) that ends a point.")
    ] = config.RALLY_BREAK_S,
    motion: Annotated[
        bool,
        typer.Option(
            "--motion",
            help="Attempt player attribution from court-half motion. Off by default: "
            "measured against hand-verified strikes it was confidently wrong. "
            "Attribution happens at labelling instead.",
        ),
    ] = False,
    show: Annotated[
        int, typer.Option("--show", help="Print the first N points.")
    ] = 8,
) -> None:
    """Stage 2: group events into points and attribute them to a player."""
    from .stages import segment as stage

    with console.status("Segmenting (scanning proxy for motion) ..."):
        pf = stage.segment(session, rally_break_s=gap, use_motion=motion)

    s = stage.summarise(pf)
    table = Table(show_header=False, box=None)
    table.add_row("points", str(s["points"]))
    table.add_row("shots", str(s["shots"]))
    table.add_row("rally length", f"median {s['median_rally_length']:.0f}, longest {s['longest_rally']}")
    table.add_row("points with a fault", str(s["faults"]))
    table.add_row(
        "attribution",
        f"near {s['near']}  far {s['far']}  unknown {s['unknown']}",
    )
    console.print(table)

    if show and pf.points:
        console.print()
        preview = Table(title=f"First {min(show, len(pf.points))} points")
        for col in ("point", "start", "shots", "fault", "server", "sequence"):
            preview.add_column(col, justify="right" if col in ("point", "shots") else "left")
        for p in pf.points[:show]:
            seq = " ".join(
                ("N" if sh.player == Player.NEAR else "F" if sh.player == Player.FAR else "?")
                for sh in p.shots
            )
            preview.add_row(
                str(p.index),
                _fmt_duration(p.t_start),
                str(len(p.shots)),
                "yes" if p.had_fault else "",
                p.server.value,
                seq,
            )
        console.print(preview)

    console.print(f"\nwrote  {config.SessionPaths(session).points}")


@app.command()
def clips(
    session: Annotated[str, typer.Argument(help="Session id (see `tennis sessions`).")],
    pre: Annotated[
        float, typer.Option("--pre", help="Seconds of lead-in before contact.")
    ] = 1.5,
    post: Annotated[
        float, typer.Option("--post", help="Seconds of follow-through after contact.")
    ] = 1.2,
    slowmo: Annotated[
        float, typer.Option("--slowmo", help="Playback slowdown, e.g. 4 for quarter speed.")
    ] = 1.0,
    all_events: Annotated[
        bool,
        typer.Option("--all", help="Export every onset, not just near-player strikes."),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Export at most N clips.")] = None,
) -> None:
    """Export one reviewable clip per detected event, with context either side."""
    from .stages import clips as stage

    with console.status("Cutting clips ..."):
        specs = stage.export(
            session,
            own_court_only=not all_events,
            pre_s=pre,
            post_s=post,
            slowmo=slowmo,
            limit=limit,
        )

    if not specs:
        console.print("[yellow]No events to export.[/] Run `tennis hits` first.")
        return

    out_dir = specs[0].path.parent
    console.print(
        f"{len(specs)} clips  ({pre + post:.1f}s each"
        + (f", {slowmo:g}x slow motion" if slowmo != 1.0 else "")
        + f", red flash at contact)"
    )
    console.print(f"folder  {out_dir}")
    console.print(f"sheet   {out_dir / 'review.csv'}")


@app.command()
def sheets(
    session: Annotated[str, typer.Argument(help="Session id.")],
    limit: Annotated[int | None, typer.Option("--limit", help="Build at most N sheets.")] = None,
    far: Annotated[
        bool, typer.Option("--far", help="Crop to the far player instead of the near one.")
    ] = False,
    crop: Annotated[
        str,
        typer.Option("--crop", help="court (whole court, robust) or player (zoomed)."),
    ] = "court",
) -> None:
    """Stage 4: build one contact sheet per shot, for the labelling model."""
    from .stages import sheets as stage

    with console.status("Building contact sheets ..."):
        results = stage.build(
            session, which=Player.FAR if far else Player.NEAR, limit=limit, crop=crop
        )

    if not results:
        console.print("[yellow]No shots.[/] Run `tennis segment` first.")
        return

    total_bytes = sum(r.path.stat().st_size for r in results)
    console.print(f"{len(results)} sheets  ({_fmt_bytes(total_bytes)} total, crop={crop})")
    if crop == "player":
        located = sum(1 for r in results if r.player_located)
        console.print(
            f"player located in {located}/{len(results)} "
            f"({located / len(results) * 100:.0f}%); the rest fall back to a fixed box"
        )
    console.print(f"folder  {results[0].path.parent}")


@app.command()
def label(
    session: Annotated[str, typer.Argument(help="Session id.")],
    action: Annotated[
        str,
        typer.Argument(help="test | submit | status | collect"),
    ] = "test",
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Stage 5: label contact sheets (event type, player, stroke) with a vision model.

    `test` runs a few immediately so you can check the prompt; `submit` sends
    everything through the Batches API at half price, which can take up to an hour.
    """
    from .stages import label as stage

    if action == "test":
        rows = stage.label_sync(session, limit=limit or 8)
        table = Table(title=f"Sample labels ({stage.MODEL})")
        for col in ("sheet", "event", "player", "stroke", "conf", "notes"):
            table.add_column(col)
        for r in rows:
            L = r.label
            table.add_row(
                r.custom_id, L["event_type"], L["player"], L["stroke"],
                L["confidence"], L["notes"][:48],
            )
        console.print(table)

    elif action == "submit":
        with console.status("Submitting batches ..."):
            ids = stage.submit(session, limit=limit)
        console.print(f"submitted {len(ids)} batch(es):")
        for i in ids:
            console.print(f"  {i}")
        console.print("\nMost batches finish within an hour. Check with:")
        console.print(f"  tennis label {session} status")

    elif action == "status":
        rows = stage.status(session)
        table = Table()
        for col in ("batch", "status", "done", "errored", "processing"):
            table.add_column(col)
        for r in rows:
            table.add_row(
                r["id"], r["status"], str(r["succeeded"]),
                str(r["errored"]), str(r["processing"]),
            )
        console.print(table)
        if all(r["status"] == "ended" for r in rows):
            console.print(f"\nAll done. Collect with:  tennis label {session} collect")

    elif action == "collect":
        with console.status("Collecting results ..."):
            summary = stage.collect(session)
        console.print(
            f"{summary['labelled']} labelled, {summary['errors']} errors\n"
            f"wrote  {summary['path']}"
        )
    else:
        console.print(f"[red]Unknown action {action!r}[/] - use test/submit/status/collect")
        raise typer.Exit(code=1)


@app.command()
def rally(
    session: Annotated[str, typer.Argument(help="Session id.")],
    point: Annotated[int, typer.Argument(help="Point index (see `tennis segment`).")],
    pre: Annotated[float, typer.Option("--pre")] = 3.0,
    post: Annotated[float, typer.Option("--post")] = 3.0,
) -> None:
    """Export one whole point as a single continuous clip."""
    from .stages import clips as stage

    with console.status(f"Cutting point {point} ..."):
        dest = stage.export_point(session, point, pre_s=pre, post_s=post)
    size = dest.stat().st_size
    console.print(f"wrote  {dest}  ({_fmt_bytes(size)})")


@app.command()
def peek(
    session: Annotated[str, typer.Argument(help="Session id (see `tennis sessions`).")],
    at: Annotated[float, typer.Option("--at", help="Timestamp in seconds.")] = 5.0,
) -> None:
    """Save a proxy frame to the session's debug folder, to sanity-check framing."""
    paths = config.SessionPaths(session)
    if not paths.proxy.exists():
        console.print(f"[red]No proxy for session {session}[/] - run `tennis ingest` first.")
        raise typer.Exit(code=1)

    dest = paths.debug / f"peek_{at:.2f}s.png"
    # The proxy is already in its final orientation, so never rotate again here.
    ffmpeg.extract_frame(paths.proxy, dest, at, autorotate=True)
    console.print(f"wrote {dest}")


@app.command()
def sessions() -> None:
    """List ingested sessions."""
    if not config.SESSIONS_DIR.exists():
        console.print("No sessions yet.")
        return

    table = Table(box=None)
    table.add_column("session id")
    table.add_column("duration", justify="right")
    table.add_column("fps", justify="right")
    table.add_column("stages")

    import json

    for d in sorted(config.SESSIONS_DIR.iterdir()):
        if not (d / "probe.json").exists():
            continue
        info = json.loads((d / "probe.json").read_text(encoding="utf-8"))
        paths = config.SessionPaths(d.name)
        done = [
            name
            for name, path in (
                ("audio", paths.audio),
                ("proxy", paths.proxy),
                ("hits", paths.hits),
                ("points", paths.points),
            )
            if path.exists()
        ]
        table.add_row(
            d.name,
            _fmt_duration(info.get("duration_s", 0)),
            f"{info.get('fps', 0):.1f}",
            " ".join(done),
        )
    console.print(table)


if __name__ == "__main__":
    app()
