"""Stage 1b: metadata extraction.

Reads embedded tags (ID3/FLAC/MP4/Vorbis/...) via mutagen where present, and
always fills in the format facts (duration, sample rate, channels, format)
from the already-probed :class:`~songanalysis.io.loader.RawAudioInfo` so the
result is complete even for files with no tags at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mutagen import File as MutagenFile

from songanalysis.io.loader import RawAudioInfo

_YEAR_RE = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class Metadata:
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    format: str
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    release_year: int | None = None
    genre: str | None = None
    track_number: int | None = None


def _first(tags, *keys: str) -> str | None:
    for key in keys:
        values = tags.get(key) if tags else None
        if values:
            value = values[0] if isinstance(values, list) else values
            text = str(value).strip()
            if text:
                return text
    return None


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    match = _YEAR_RE.search(value)
    return int(match.group(1)) if match else None


def _parse_track_number(value: str | None) -> int | None:
    if not value:
        return None
    # Common formats: "3", "3/12"
    head = value.split("/", 1)[0].strip()
    match = re.match(r"\d+", head)
    return int(match.group(0)) if match else None


def extract_metadata(path: str | Path, raw: RawAudioInfo) -> Metadata:
    """Extract embedded tags plus raw format facts. Never raises on missing
    or malformed tags -- absence is represented as ``None`` fields."""
    path = Path(path)

    artist = title = album = genre = None
    year: int | None = None
    track_number: int | None = None

    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception:
        audio = None

    if audio is not None and getattr(audio, "tags", None):
        tags = audio.tags
        artist = _first(tags, "artist", "albumartist", "performer")
        title = _first(tags, "title")
        album = _first(tags, "album")
        genre = _first(tags, "genre")
        year = _parse_year(_first(tags, "date", "originaldate", "year"))
        track_number = _parse_track_number(_first(tags, "tracknumber", "track"))

    return Metadata(
        filename=path.name,
        duration_sec=raw.duration_sec,
        sample_rate=raw.sample_rate,
        channels=raw.channels,
        format=raw.format,
        artist=artist,
        title=title,
        album=album,
        release_year=year,
        genre=genre,
        track_number=track_number,
    )
