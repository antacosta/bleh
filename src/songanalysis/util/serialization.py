"""Helpers for turning dataclasses / numpy-heavy structures into plain,
JSON-safe Python objects (and back into JSON text)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, numpy arrays/scalars, tuples, etc.
    into plain ``dict``/``list``/``float``/``int``/``str``/``None`` values
    that :func:`json.dumps` can handle natively and deterministically."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(obj, 6)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_jsonable(obj), indent=indent, sort_keys=False, allow_nan=False)
