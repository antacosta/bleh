"""Exceptions raised by the analysis pipeline. All are subclasses of
:class:`SongAnalysisError` so callers can catch broadly or narrowly."""

from __future__ import annotations


class SongAnalysisError(Exception):
    """Base class for all errors raised by songanalysis."""


class UnsupportedAudioFileError(SongAnalysisError):
    """The input file could not be decoded as audio (missing, corrupt, or
    an unsupported/unrecognized format)."""


class EmptyAudioError(SongAnalysisError):
    """The decoded audio has zero (or effectively zero) frames."""
