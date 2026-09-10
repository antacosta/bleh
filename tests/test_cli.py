from __future__ import annotations

import json

from songanalysis import cli


def test_analyze_prints_valid_json(tmp_path, click_track_path, capsys):
    exit_code = cli.main(["analyze", click_track_path, "--cache-dir", str(tmp_path / "cache")])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "analysis_version" in data
    assert "dj_affordances" in data


def test_analyze_writes_to_output_file(tmp_path, click_track_path, capsys):
    out_path = tmp_path / "result.json"
    exit_code = cli.main(
        ["analyze", click_track_path, "--cache-dir", str(tmp_path / "cache"), "-o", str(out_path)]
    )
    assert exit_code == 0
    data = json.loads(out_path.read_text())
    assert "metadata" in data


def test_inspect_prints_human_readable_summary(tmp_path, structured_song_path, capsys):
    exit_code = cli.main(["inspect", structured_song_path, "--cache-dir", str(tmp_path / "cache")])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Song:" in out
    assert "BPM:" in out
    assert "Structure:" in out
    assert "Mix-in candidates:" in out
    assert "Mix-out candidates:" in out
    assert "Loops:" in out
    assert "Major events:" in out


def test_analyze_missing_file_returns_error_exit_code(tmp_path, missing_file_path, capsys):
    exit_code = cli.main(["analyze", missing_file_path, "--cache-dir", str(tmp_path / "cache")])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error:" in err


def test_analyze_malformed_file_returns_error_exit_code(tmp_path, malformed_file_path, capsys):
    exit_code = cli.main(["analyze", malformed_file_path, "--cache-dir", str(tmp_path / "cache")])
    assert exit_code == 1


def test_no_cache_flag_bypasses_cache_dir(tmp_path, click_track_path, capsys):
    cache_dir = tmp_path / "cache"
    exit_code = cli.main(["analyze", click_track_path, "--cache-dir", str(cache_dir), "--no-cache"])
    assert exit_code == 0
    assert not cache_dir.exists() or list(cache_dir.glob("*.json")) == []
