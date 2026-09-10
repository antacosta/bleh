from __future__ import annotations

import shutil
import subprocess

import pytest

from songanalysis.io.loader import load_audio
from songanalysis.io.metadata import extract_metadata


def test_metadata_with_no_tags_falls_back_to_format_facts(click_track_path):
    buf = load_audio(click_track_path)
    meta = extract_metadata(buf.path, buf.raw)

    assert meta.filename.endswith(".wav")
    assert meta.duration_sec > 0
    assert meta.sample_rate > 0
    assert meta.channels == 1
    assert meta.format
    # No embedded tags in a plain synthesized WAV -> everything else is None,
    # not fabricated.
    assert meta.artist is None
    assert meta.title is None
    assert meta.album is None
    assert meta.release_year is None
    assert meta.genre is None
    assert meta.track_number is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg to build a tagged fixture file")
def test_metadata_with_embedded_tags(tmp_path):
    mp3_path = tmp_path / "tagged.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-metadata",
            "artist=Test Artist",
            "-metadata",
            "title=Test Title",
            "-metadata",
            "album=Test Album",
            "-metadata",
            "date=2019",
            "-metadata",
            "genre=House",
            "-metadata",
            "track=3/10",
            str(mp3_path),
            "-loglevel",
            "error",
        ],
        check=True,
    )

    buf = load_audio(mp3_path)
    meta = extract_metadata(buf.path, buf.raw)

    assert meta.artist == "Test Artist"
    assert meta.title == "Test Title"
    assert meta.album == "Test Album"
    assert meta.release_year == 2019
    assert meta.genre == "House"
    assert meta.track_number == 3
    assert meta.duration_sec == pytest.approx(2.0, abs=0.2)
