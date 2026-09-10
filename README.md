# songanalysis

A local, offline song-analysis layer for an AI DJ system. Given an audio
file, it produces a deterministic, cacheable, structured JSON analysis
(tempo/beat grid, key/harmony, an energy timeline, song structure, vocal
activity, and DJ-relevant mix-in/mix-out/loop/event affordances) for a
future planning layer to consume.

This project deliberately stops at analysis. It does not recommend songs,
score candidates, plan transitions, mix, render audio, or use any cloud/LLM
service — see "Scope" below.

## Install

```
uv sync
```

Requires `ffmpeg` on `PATH` for a few less-common audio containers
(most WAV/FLAC/MP3/OGG files are decoded directly via libsndfile and don't
need it).

## CLI

```
uv run songanalysis analyze  path/to/song.mp3          # canonical JSON -> stdout
uv run songanalysis analyze  path/to/song.mp3 -o out.json
uv run songanalysis inspect  path/to/song.mp3           # human-readable summary
```

Useful flags (both subcommands): `--no-cache`, `--force`, `--cache-dir DIR`,
`-v/--verbose` (prints cache hit/miss to stderr).

Results are cached on disk (default `~/.cache/songanalysis`, override with
`SONGANALYSIS_CACHE_DIR` or `--cache-dir`), keyed by the file's content hash
and the analysis schema's version, so re-analyzing the same file is
near-instant.

## Architecture

```
io.loader/io.metadata          -- audio decoding + tag/format extraction
features.loudness/tempo_beat/  -- independent feature-extraction modules,
  harmonic/energy/vocals          each a plain function over (y, sr) -> dataclass
structure.segmentation         -- boundary detection + section/repetition analysis
dj.affordances                 -- mix-in/out points, loop candidates, events
schema.py                      -- SongAnalysis -> versioned JSON shape
cache.py                       -- content-hash disk cache
pipeline.py                    -- wires the stages together
cli.py                         -- `analyze` / `inspect` commands
```

Each feature module is independently testable on a plain numpy array; the
pipeline only sequences them and threads outputs that later stages reuse
(e.g. the onset envelope computed once for beat tracking is reused for the
energy timeline instead of being recomputed).

## Confidence and heuristics

Nothing is asserted with fabricated certainty. Every detected property that
can be wrong (BPM, key, time signature, downbeats, section labels, vocal
activity, DJ affordances) carries an explicit confidence value or string
(`"measured"` / `"heuristic"` / `"insufficient_data"` / `"silent"`), and
degrades to `None`/empty rather than guessing when the signal is too weak
or the audio too short. Structural section boundaries are always reported;
the semantic label (`"chorus"`, `"intro"`, ...) is a separate, lower-confidence
guess layered on top of the neutral `section_N` id.

## Scope

Explicitly out of scope for this layer (belongs to a later planning/mixing
layer): song recommendation, candidate scoring, playlist generation,
transition/crossfade selection, beatmatching, audio rendering, and any
LLM/cloud service.
