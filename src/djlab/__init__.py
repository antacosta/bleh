"""AI DJ Lab -- a local developer/testing UI around the existing engine.

This package is a thin orchestration/adapter layer only. It calls
``songanalysis`` (analysis), ``songscoring`` (one-step compatibility
scoring), and ``songplanner`` (look-ahead sequence planning) through their
existing public interfaces and reshapes their outputs into plain JSON for a
local browser UI -- it does not reanalyze audio, rescore candidates, replan
sequences, or otherwise duplicate any algorithmic logic from those layers.
Transition Selection and Audio Rendering do not exist yet; this package
reflects that honestly (see ``planning.py``) rather than fabricating a
stand-in.
"""
