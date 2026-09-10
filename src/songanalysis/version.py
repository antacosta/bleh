"""Schema/analysis versioning.

``ANALYSIS_VERSION`` is embedded in every analysis result and in the cache
key. Bump it whenever the *meaning* of an existing field changes, a field is
removed, or the feature-extraction algorithm for an existing field changes
in a way that would make old and new results inconsistent. Purely additive
changes (new optional fields) can keep the same MAJOR but should still bump
MINOR so downstream code can detect "this cache entry predates field X".

Format: "MAJOR.MINOR". Cached analyses whose stored version's MAJOR differs
from the current MAJOR are treated as stale and re-computed.
"""

ANALYSIS_VERSION = "1.0"


def parse_major(version: str) -> str:
    return version.split(".", 1)[0]


def is_compatible(cached_version: str) -> bool:
    """Whether a cached analysis produced under ``cached_version`` can still
    be served as-is under the current :data:`ANALYSIS_VERSION`."""
    return parse_major(cached_version) == parse_major(ANALYSIS_VERSION)
