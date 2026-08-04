"""Typed domain errors used by generators, validators and adapters."""


class SpatialEpisodeError(Exception):
    """Base class for expected project failures."""


class FrameMismatchError(SpatialEpisodeError, ValueError):
    """Raised when transforms with incompatible frames are composed."""


class CapabilityError(SpatialEpisodeError, ValueError):
    """Raised when a recipe asks a source for unsupported supervision."""


class GraphValidationError(SpatialEpisodeError, ValueError):
    """Raised when an operation graph is cyclic, untyped or incomplete."""
