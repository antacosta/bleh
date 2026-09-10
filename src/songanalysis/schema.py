"""The top-level analysis result and its JSON schema shape.

``SongAnalysis`` holds the full, structured output of every stage as plain
dataclasses (so it stays a normal, typed Python object to work with in
code). ``to_schema_dict`` renders that into the versioned, JSON-serializable
shape described in the project's README -- the "conceptual schema" from the
spec, adapted to this repo's per-stage dataclasses rather than hand-built
dicts, so every field a feature module computes is preserved automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from songanalysis.features.energy import EnergyTimeline
from songanalysis.features.harmonic import HarmonicFeatures
from songanalysis.features.loudness import LoudnessFeatures
from songanalysis.features.tempo_beat import TempoBeatFeatures
from songanalysis.features.vocals import VocalTimeline
from songanalysis.io.metadata import Metadata
from songanalysis.dj.affordances import DJAffordances
from songanalysis.structure.segmentation import StructureAnalysis
from songanalysis.util.serialization import to_jsonable
from songanalysis.version import ANALYSIS_VERSION


@dataclass(frozen=True)
class GlobalMusicalProperties:
    """Single-value summary of the song's overall musical character.

    Full time-varying detail backing these numbers lives in ``harmonic``,
    ``rhythm``, and ``timeline.energy`` -- this section is a convenience
    rollup, not a separate source of truth.
    """

    duration_sec: float
    bpm: float | None
    bpm_confidence: float
    time_signature: str | None
    time_signature_confidence: float
    key: str | None
    key_confidence: float
    integrated_lufs: float | None
    integrated_lufs_confidence: str
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    loudness_range_db: float | None
    loudness_range_confidence: str
    overall_energy: float
    energy_trend: float


@dataclass(frozen=True)
class SongAnalysis:
    analysis_version: str
    metadata: Metadata
    loudness: LoudnessFeatures
    tempo_beat: TempoBeatFeatures
    harmonic: HarmonicFeatures
    energy_timeline: EnergyTimeline
    vocal_timeline: VocalTimeline
    structure: StructureAnalysis
    dj_affordances: DJAffordances

    @property
    def global_properties(self) -> GlobalMusicalProperties:
        return GlobalMusicalProperties(
            duration_sec=self.metadata.duration_sec,
            bpm=self.tempo_beat.bpm,
            bpm_confidence=self.tempo_beat.bpm_confidence,
            time_signature=self.tempo_beat.time_signature,
            time_signature_confidence=self.tempo_beat.time_signature_confidence,
            key=self.harmonic.key,
            key_confidence=self.harmonic.key_confidence,
            integrated_lufs=self.loudness.integrated_lufs,
            integrated_lufs_confidence=self.loudness.integrated_lufs_confidence,
            peak_dbfs=self.loudness.peak_dbfs,
            rms_dbfs=self.loudness.rms_dbfs,
            crest_factor_db=self.loudness.crest_factor_db,
            loudness_range_db=self.loudness.loudness_range_db,
            loudness_range_confidence=self.loudness.loudness_range_confidence,
            overall_energy=self.energy_timeline.overall_energy,
            energy_trend=self.energy_timeline.energy_trend,
        )

    def to_schema_dict(self) -> dict:
        return {
            "analysis_version": self.analysis_version,
            "metadata": self.metadata,
            "global": self.global_properties,
            "timeline": {
                "beats": self.tempo_beat.beat_times,
                "downbeats": self.tempo_beat.downbeat_times,
                "onsets": self.tempo_beat.onset_times,
                "energy": self.energy_timeline,
                "vocals": {
                    "regions": self.vocal_timeline.regions,
                    "times": self.vocal_timeline.times,
                    "probability": self.vocal_timeline.vocal_probability,
                    "intensity": self.vocal_timeline.vocal_intensity,
                    "density": self.vocal_timeline.vocal_density,
                    "method": self.vocal_timeline.method,
                    "confidence": self.vocal_timeline.confidence,
                },
            },
            "structure": self.structure,
            "harmonic": self.harmonic,
            "rhythm": self.tempo_beat,
            "dj_affordances": self.dj_affordances,
        }

    def to_jsonable(self) -> dict:
        return to_jsonable(self.to_schema_dict())
