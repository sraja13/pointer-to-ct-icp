"""Data models for PA3 matching phase."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RigidBody:
    """Represents a rigid body with markers and a tip point."""
    markers: np.ndarray  # shape (N, 3)
    tip: np.ndarray  # shape (3,)


@dataclass
class SampleFrame:
    """Represents a single sample frame with markers from two rigid bodies."""
    markers_a: np.ndarray  # shape (NA, 3)
    markers_b: np.ndarray  # shape (NB, 3)


@dataclass
class MatchResult:
    """Represents the result of matching a sample point to the mesh."""
    sample_point: np.ndarray
    closest_point: np.ndarray
    distance: float

