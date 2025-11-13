#!/usr/bin/env python3
"""
PA3 matching phase for a simplified ICP pipeline.

Goes through the provided mesh, rigid body definition, and sample readings
files to compute the pointer tip position for each sample frame and locate the
closest point on the mesh surface. Results are written in the format prescribed
for PA3 output files.
"""

# Rport all public functions for backward compatibility
# Both relative imports (when run as module) and absolute imports (when imported directly)
try:
    from .cli import main
    from .geometry import closest_point_on_mesh, closest_point_on_triangle
    from .io import load_mesh, load_rigid_body, load_samples
    from .matching import compute_matches
    from .models import MatchResult, RigidBody, SampleFrame
    from .output import format_output_line
    from .transforms import invert_transform, point_cloud_registration, transform_point
except ImportError:
    # Fall back to absolute imports when imported directly (e.g., in tests)
    from cli import main
    from geometry import closest_point_on_mesh, closest_point_on_triangle
    from io import load_mesh, load_rigid_body, load_samples
    from matching import compute_matches
    from models import MatchResult, RigidBody, SampleFrame
    from output import format_output_line
    from transforms import invert_transform, point_cloud_registration, transform_point

__all__ = [
    "main",
    "RigidBody",
    "SampleFrame",
    "MatchResult",
    "load_mesh",
    "load_rigid_body",
    "load_samples",
    "point_cloud_registration",
    "invert_transform",
    "transform_point",
    "closest_point_on_triangle",
    "closest_point_on_mesh",
    "compute_matches",
    "format_output_line",
]

if __name__ == "__main__":
    main()

