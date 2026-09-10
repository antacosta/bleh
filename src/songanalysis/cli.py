"""Command-line interface.

    songanalysis analyze  <file>   -> canonical JSON analysis (stdout or -o file)
    songanalysis inspect  <file>   -> human-readable summary

Both share the same caching pipeline; ``inspect`` is a formatting layer over
the exact same result ``analyze`` produces, never a separate computation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from songanalysis.cache import AnalysisCache, analyze_song_cached
from songanalysis.errors import SongAnalysisError
from songanalysis.io.loader import DEFAULT_ANALYSIS_SR
from songanalysis.pipeline import analyze_song


def _seconds_to_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def format_human_summary(result: dict) -> str:
    meta = result.get("metadata", {})
    glob = result.get("global", {})
    structure = result.get("structure", {})
    dj = result.get("dj_affordances", {})

    title = meta.get("title") or meta.get("filename", "unknown")
    artist = meta.get("artist")
    song_line = f"{artist} - {title}" if artist else title

    lines = []
    lines.append(f"Song:      {song_line}")
    lines.append(f"Duration:  {_seconds_to_mmss(meta.get('duration_sec', 0))}")

    bpm = glob.get("bpm")
    bpm_conf = glob.get("bpm_confidence", 0.0)
    ts = glob.get("time_signature") or "unknown"
    lines.append(f"BPM:       {bpm:.1f} (confidence {_fmt_pct(bpm_conf)}), time signature {ts}" if bpm else "BPM:       unknown")

    key = glob.get("key") or "unknown"
    key_conf = glob.get("key_confidence", 0.0)
    lines.append(f"Key:       {key} (confidence {_fmt_pct(key_conf)})")

    lufs = glob.get("integrated_lufs")
    lufs_str = f"{lufs:.1f} LUFS" if lufs is not None else f"unmeasured ({glob.get('integrated_lufs_confidence')})"
    lines.append(f"Loudness:  {lufs_str}, peak {glob.get('peak_dbfs', 0):.1f} dBFS")
    lines.append(f"Energy:    overall {glob.get('overall_energy', 0):.2f}/1.0, trend {glob.get('energy_trend', 0):+.4f}/s")

    lines.append("Structure:")
    for section in structure.get("sections", []):
        label = section.get("heuristic_label") or section["id"]
        conf = section.get("heuristic_label_confidence", 0.0)
        label_str = f"{label} ({_fmt_pct(conf)})" if section.get("heuristic_label") else label
        lines.append(
            f"  {_seconds_to_mmss(section['start'])}-{_seconds_to_mmss(section['end'])}  {label_str:20s} "
            f"energy={section['energy']:.2f} vocal={_fmt_pct(section['vocal_presence'])}"
        )

    lines.append("Mix-in candidates:")
    for p in dj.get("mix_in_points", []):
        lines.append(f"  {_seconds_to_mmss(p['time'])}  ({_fmt_pct(p['confidence'])})  {', '.join(p['reasons'])}")
    if not dj.get("mix_in_points"):
        lines.append("  (none found)")

    lines.append("Mix-out candidates:")
    for p in dj.get("mix_out_points", []):
        lines.append(f"  {_seconds_to_mmss(p['time'])}  ({_fmt_pct(p['confidence'])})  {', '.join(p['reasons'])}")
    if not dj.get("mix_out_points"):
        lines.append("  (none found)")

    loops = dj.get("loop_candidates", [])
    lines.append(f"Loops:     {len(loops)} candidate(s)")
    for loop in loops[:8]:
        lines.append(
            f"  {_seconds_to_mmss(loop['start'])}-{_seconds_to_mmss(loop['end'])}  {loop['bars']} bars "
            f"({_fmt_pct(loop['confidence'])})"
        )

    events = dj.get("events", [])
    lines.append(f"Major events:")
    for e in events:
        detail = ", ".join(f"{k}={v}" for k, v in (e.get("details") or {}).items())
        lines.append(f"  {_seconds_to_mmss(e['time'])}  {e['type']:20s} ({_fmt_pct(e['confidence'])})  {detail}")
    if not events:
        lines.append("  (none found)")

    return "\n".join(lines)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="Path to a local audio file")
    parser.add_argument("--no-cache", action="store_true", help="Do not read from the analysis cache")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite the cache entry")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the cache directory")
    parser.add_argument(
        "--sr", type=int, default=DEFAULT_ANALYSIS_SR, dest="analysis_sr", help="Analysis sample rate (advanced)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print cache hit/miss info to stderr")


def _run_analysis(args: argparse.Namespace) -> dict:
    if args.no_cache:
        result = analyze_song(args.path, analysis_sr=args.analysis_sr).to_jsonable()
        was_cached = False
    else:
        cache = AnalysisCache(cache_dir=args.cache_dir)
        result, was_cached = analyze_song_cached(
            args.path, cache=cache, force=args.force, analysis_sr=args.analysis_sr
        )
    if getattr(args, "verbose", False):
        print(f"[{'cache hit' if was_cached else 'computed'}] {args.path}", file=sys.stderr)
    return result


def cmd_analyze(args: argparse.Namespace) -> int:
    result = _run_analysis(args)
    text = json.dumps(result, indent=None if args.compact else 2)
    if args.output:
        args.output.write_text(text)
        print(f"Wrote analysis to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    result = _run_analysis(args)
    print(format_human_summary(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="songanalysis", description="Song-analysis layer for an AI DJ system")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print cache hit/miss info to stderr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Produce the canonical machine-readable JSON analysis")
    _add_common_args(analyze_parser)
    analyze_parser.add_argument("-o", "--output", type=Path, default=None, help="Write JSON to a file instead of stdout")
    analyze_parser.add_argument("--compact", action="store_true", help="Emit compact (non-indented) JSON")
    analyze_parser.set_defaults(func=cmd_analyze)

    inspect_parser = subparsers.add_parser("inspect", help="Print a human-readable analysis summary")
    _add_common_args(inspect_parser)
    inspect_parser.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SongAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
