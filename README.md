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

---

# songscoring

A separate, independent layer (`src/songscoring/`) that consumes
`songanalysis` JSON and answers one question: *given the song currently
playing, how good would each other song in the library be as the next one?*
It does not reanalyze audio, does not pick or perform a transition, and does
not do multi-song look-ahead planning -- see `songscoring/scorer.py`'s
module docstring and the architecture diagram below.

## Usage

```python
from songscoring.song_profile import profile_from_json_file
from songscoring.state import DJState
from songscoring.scorer import rank_candidates

current = profile_from_json_file("now_playing.json")
library = [profile_from_json_file(p) for p in other_analysis_files]

state = DJState(current_song=current, current_position=180.0)
ranking = rank_candidates(state, library)

for r in ranking[:10]:
    print(r.candidate_id, r.total_score, r.to_dict())
```

## Architecture

```
audio -> songanalysis -> SongProfile (songscoring/song_profile.py)
                              |
   DJState (songscoring/state.py) ---> score_candidate() per component:
                              |          tempo, harmonic, energy, rhythm,
                              |          structure, style, familiarity,
                              |          repetition  (songscoring/components/)
                              v
                    confidence-weighted total_score
                    (songscoring/scorer.py, weights in config.py)
                              |
                              v
                      rank_candidates() -> CandidateScore list
```

Every scoring component is a pure function of `(current_song, candidate,
state, config) -> ComponentScore`, independently unit-tested. All weights
live in `songscoring/config.py`, never hard-coded in a component.

## Confidence-aware scoring

Every component reports a `value` (0..100, its best-guess assessment) and a
separate `confidence` (0..1, how much to trust that assessment) -- these are
never collapsed into one number. The central combiner
(`scorer.compute_total_score`) scales each component's *weight* by its own
confidence before averaging, so a component that's unsure barely moves the
total, while its `value` stays visible for inspection.

## Explicit energy intent

`DJState.desired_energy_direction` can be stated at three distinguishable
strengths (`EnergyIntentStrength`: `NONE` / `MILD` / `EXPLICIT`, see
`state.py`). An `EXPLICIT` request (the default once a direction is set)
gives the energy component substantially more influence than its base
weight -- via the *same* confidence-scaling mechanism `compute_total_score`
already uses, generalized with an optional `weight_multipliers` argument,
not a second, special-cased formula (see `config.EnergyIntentConfig` and
`scorer.energy_intent_multiplier`). With no explicit direction, scoring is
unchanged from the original balanced behavior.

---

# songplanner

A third, independent layer (`src/songplanner/`) that plans a short
*sequence* of upcoming songs (~3-5), not just the single best next one, via
bounded beam search. It consumes `songscoring` to prune candidates and
score transitions -- it does not reanalyze audio, duplicate scoring logic,
or choose/render an actual transition (crossfade, beatmatching, stems);
that is a later layer's job. See `songplanner/planner.py`'s module
docstring.

## Usage

```python
from songscoring.song_profile import profile_from_json_file
from songscoring.state import DJState, EnergyDirection
from songplanner.planner import plan_sequence

current = profile_from_json_file("now_playing.json")
library = [profile_from_json_file(p) for p in other_analysis_files]

state = DJState(current_song=current, desired_energy_direction=EnergyDirection.INCREASE)
plan = plan_sequence(state, library)

for step in plan.steps:
    print(step.position, step.song_id, step.transition_score.total_score)
print(plan.sequence_score.to_dict())    # full structured breakdown + reasons
print(plan.local_vs_global)             # why the chosen opening beat the alternatives
```

## Architecture

```
audio -> songanalysis -> SongProfile -> songscoring (one-step scorer)
                                              |
                              PlannerState (planner_state.py): a
                              hypothetical, immutable DJState + the
                              sequence chosen so far -- advance() returns
                              a new state, never mutates the real one
                                              |
                              candidate_pool.build_pool(): rank_candidates()
                              prunes the library to a bounded pool at each
                              node (not an exhaustive per-depth scan)
                                              |
                              beam search (planner.py): generate -> score
                              whole paths -> keep best beam_width -> expand
                              -> repeat for horizon steps
                                              |
                              sequence_scoring.score_sequence(): combines
                              mean transition score with energy-trajectory,
                              variety, and coherence -- via the *same*
                              compute_total_score combiner songscoring uses
                                              v
                                      Plan (types.py): steps + full
                                      structured score breakdown +
                                      local-vs-global comparison
```

`[future layers, not built here]`: Transition Selection (how to actually
mix from one planned song into the next) -> Audio Rendering.

## Why look-ahead, not just repeated one-step scoring

A purely greedy one-step scorer can only ask "how good is B after A". The
planner additionally asks "how good is B after A, given where B lets us go
next" -- so it can prefer A -> B -> C over A -> D -> E even when A -> D
scores higher alone, if B opens onto a much better C. This requires
comparing whole hypothetical paths, which is exactly what beam search
(not a single ranking pass) provides: multiple candidate paths survive
each step, and the *final* choice is decided by total sequence score, not
by which move looked best immediately.

## Sequence-level scoring, beyond a sum of transition scores

`sequence_scoring.score_sequence` combines four components -- via the
identical confidence-weighted combiner `songscoring.scorer.
compute_total_score` uses for one-step scoring, not a second formula:

- `transition`: mean of the path's own one-step scores (already computed;
  reused, not redone).
- `trajectory`: does the path's energy shape match the requested direction
  (BUILD/RELEASE/MAINTAIN), or read as a coherent shape with no direction
  at all -- rewarding an overall trend, not strict monotonicity, and
  penalizing RELEASE's "instant collapse" as a distinct failure mode.
- `variety`: flat, non-decaying penalty for a song/artist/genre recurring
  within the plan (the one-step repetition component's recency-decay isn't
  strict enough for one short plan; see config.py).
- `coherence`: penalizes erratic genre "bouncing" (A -> B -> A) while
  allowing progression (A -> B -> C) or a sustained run (A -> A -> B).

`trajectory`'s *weight* is scaled by the same `EnergyIntentConfig`
multiplier the one-step energy-intent fix uses, and its *confidence* tracks
the per-step energy component's confidence specifically -- so an uncertain
energy read dampens the trajectory claim without needing a second
uncertainty model.

## Explainability

`Plan.to_dict()` / `SequenceScoreBreakdown.to_dict()` expose per-transition
scores, sequence-level component values, the energy-level trajectory, and a
`reasons` list -- all built by formatting numbers already computed
elsewhere in this package. Nothing in `songplanner` (or any layer here)
generates natural-language explanations via an LLM.
