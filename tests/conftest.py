from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 22050


def sine(freq: float, duration: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_track(duration: float, bpm: float, sr: int = SR, amplitude: float = 0.6, accent_every: int = 4) -> np.ndarray:
    """A percussive click track with a clear beat grid and 4-on-the-floor
    accent pattern, plus a quiet harmonic bed so it isn't pure silence
    between clicks."""
    beat_period = 60.0 / bpm
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float64)
    click_len = 300
    beat_idx = 0
    t_cursor = 0.0
    while t_cursor < duration:
        idx = int(t_cursor * sr)
        if idx < n - click_len:
            accent = 1.0 if beat_idx % accent_every == 0 else 0.55
            y[idx : idx + click_len] += amplitude * accent * np.hanning(click_len)
        t_cursor += beat_period
        beat_idx += 1
    t = np.arange(n) / sr
    y += 0.05 * np.sin(2 * np.pi * 220 * t)
    return y.astype(np.float32)


def am_vocal_tone(duration: float, sr: int = SR, carrier: float = 500.0, mod_hz: float = 4.0, amplitude: float = 0.3, seed: int = 0) -> np.ndarray:
    """A crude stand-in for a vocal: a mid-band carrier amplitude-modulated
    at the syllabic rate (~4 Hz) our vocal heuristic looks for."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    phase = rng.uniform(0, 2 * np.pi)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * mod_hz * t + phase)
    return (amplitude * np.sin(2 * np.pi * carrier * t) * am).astype(np.float32)


def silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration * sr), dtype=np.float32)


def write_wav(path, y: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), y, sr)
    return str(path)


def build_structured_song(sr: int = SR) -> np.ndarray:
    """~50s song with a clear intro / verse / chorus / verse / chorus / outro
    shape: quiet instrumental bookends, alternating vocal-present sections,
    and a louder repeated "chorus" section, used for structure/DJ tests."""

    def section(dur, bpm, bass_amp, vocal_amp, seed):
        beat = click_track(dur, bpm, sr=sr, amplitude=bass_amp)
        if vocal_amp > 0:
            beat = beat + am_vocal_tone(dur, sr=sr, amplitude=vocal_amp, seed=seed)
        return beat

    intro = section(8, 120, 0.15, 0.0, 1)
    verse1 = section(10, 120, 0.3, 0.2, 2)
    chorus1 = section(8, 120, 0.5, 0.3, 3)
    verse2 = section(10, 120, 0.3, 0.2, 4)
    chorus2 = section(8, 120, 0.5, 0.3, 5)
    outro = section(6, 120, 0.1, 0.0, 6)
    return np.concatenate([intro, verse1, chorus1, verse2, chorus2, outro]).astype(np.float32)


@pytest.fixture(scope="session")
def structured_song_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "structured_song.wav"
    write_wav(path, build_structured_song())
    return str(path)


@pytest.fixture(scope="session")
def click_track_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "click_track.wav"
    write_wav(path, click_track(20.0, 128.0))
    return str(path)


@pytest.fixture()
def silent_wav_path(tmp_path):
    path = tmp_path / "silent.wav"
    write_wav(path, silence(5.0))
    return str(path)


@pytest.fixture()
def very_short_wav_path(tmp_path):
    path = tmp_path / "short.wav"
    write_wav(path, sine(440.0, 0.05))
    return str(path)


@pytest.fixture()
def malformed_file_path(tmp_path):
    path = tmp_path / "malformed.wav"
    path.write_bytes(b"this is not a valid audio file, just text pretending to be one")
    return str(path)


@pytest.fixture()
def empty_file_path(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture()
def missing_file_path(tmp_path):
    return str(tmp_path / "does_not_exist.wav")
