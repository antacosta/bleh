"""Stage 1: audio loading.

Turns a path on disk into an :class:`AudioBuffer` — a decoded, mono,
analysis-rate waveform plus the *original* file's raw format facts (sample
rate, channel count, duration, container format) captured before any
resampling. Downstream feature-extraction stages only depend on
``AudioBuffer``, never on file I/O directly, so they stay testable on
synthetic arrays.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from songanalysis.errors import EmptyAudioError, UnsupportedAudioFileError

#: Default sample rate audio is resampled to for analysis. 22.05 kHz is
#: standard in MIR work (librosa's default) and captures everything below
#: ~11 kHz, which is where essentially all rhythmic/harmonic/energy
#: information relevant to DJ-style analysis lives. Loudness/peak
#: measurements are taken from this same signal for consistency; anyone
#: needing full-bandwidth loudness can re-run with a higher target rate.
DEFAULT_ANALYSIS_SR = 22050


@dataclass(frozen=True)
class RawAudioInfo:
    """Facts about the *original* file, independent of analysis resampling."""

    sample_rate: int
    channels: int
    duration_sec: float
    format: str
    subtype: str | None = None


@dataclass(frozen=True)
class AudioBuffer:
    """Decoded mono audio ready for feature extraction."""

    y: np.ndarray  # mono, float32, at `sr`
    sr: int
    path: Path
    raw: RawAudioInfo

    @property
    def duration_sec(self) -> float:
        return len(self.y) / self.sr if self.sr else 0.0

    @property
    def is_effectively_silent(self) -> bool:
        if self.y.size == 0:
            return True
        return bool(np.max(np.abs(self.y)) < 1e-6)


def _probe_with_soundfile(path: Path) -> RawAudioInfo | None:
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return RawAudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        duration_sec=float(duration),
        format=str(info.format),
        subtype=str(info.subtype) if info.subtype else None,
    )


def _probe_with_ffprobe(path: Path) -> RawAudioInfo | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-select_streams",
                "a:0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        payload = json.loads(proc.stdout)
        stream = (payload.get("streams") or [{}])[0]
        fmt = payload.get("format", {})
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
        if sample_rate <= 0 or channels <= 0:
            return None
        return RawAudioInfo(
            sample_rate=sample_rate,
            channels=channels,
            duration_sec=duration,
            format=str(fmt.get("format_name", "unknown")),
            subtype=str(stream.get("codec_name")) if stream.get("codec_name") else None,
        )
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def probe_audio_file(path: Path) -> RawAudioInfo:
    """Read container-level facts about ``path`` without fully decoding it.

    Tries libsndfile first (fast, no subprocess), then falls back to
    ``ffprobe`` for formats libsndfile doesn't handle (e.g. some AAC/M4A/WMA
    files). Raises :class:`UnsupportedAudioFileError` if neither works.
    """
    info = _probe_with_soundfile(path) or _probe_with_ffprobe(path)
    if info is None:
        raise UnsupportedAudioFileError(f"Could not read audio format info for {path}")
    return info


def load_audio(
    path: str | Path,
    *,
    analysis_sr: int = DEFAULT_ANALYSIS_SR,
) -> AudioBuffer:
    """Decode ``path`` to a mono waveform at ``analysis_sr`` for feature
    extraction, alongside the original file's raw format facts.

    Raises:
        UnsupportedAudioFileError: file missing, corrupt, or undecodable.
        EmptyAudioError: file decodes but contains zero audio frames.
    """
    path = Path(path)
    if not path.is_file():
        raise UnsupportedAudioFileError(f"No such file: {path}")

    raw = probe_audio_file(path)

    try:
        y, sr = librosa.load(str(path), sr=analysis_sr, mono=True)
    except Exception as exc:  # librosa/audioread/soundfile raise varied types
        raise UnsupportedAudioFileError(f"Could not decode audio file {path}: {exc}") from exc

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise EmptyAudioError(f"Decoded zero audio frames from {path}")

    if raw.duration_sec <= 0.0:
        raw = RawAudioInfo(
            sample_rate=raw.sample_rate or sr,
            channels=raw.channels or 1,
            duration_sec=len(y) / sr,
            format=raw.format,
            subtype=raw.subtype,
        )

    return AudioBuffer(y=y, sr=int(sr), path=path, raw=raw)
