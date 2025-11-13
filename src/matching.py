"""Main matching computation logic for PA3."""

from typing import List

import numpy as np

from typing import Dict, Optional

from .geometry import build_triangle_accelerator, closest_point_on_mesh
from .models import MatchResult, RigidBody, SampleFrame
from .transforms import invert_transform, point_cloud_registration, transform_point


def compute_matches(
    vertices: np.ndarray,
    triangles: np.ndarray,
    body_a: RigidBody,
    body_b: RigidBody,
    samples: List[SampleFrame],
    mesh_accel: Optional[Dict[str, object]] = None,
) -> List[MatchResult]:
    """
    Compute closest point matches for all sample frames.
    
    For each sample frame:
    1. Register body A markers to sample markers A
    2. Register body B markers to sample markers B
    3. Transform body A tip through the coordinate systems
    4. Find closest point on mesh to the transformed tip
    
    Args:
        vertices: Mesh vertices, shape (N, 3)
        triangles: Triangle vertex indices, shape (M, 3)
        body_a: Rigid body A definition
        body_b: Rigid body B definition
        samples: List of sample frames to process
    
    Returns:
        List of MatchResult objects, one per sample frame
    """
    identity_registration = np.eye(4, dtype=float)

    accel = mesh_accel or build_triangle_accelerator(vertices, triangles)

    results: List[MatchResult] = []
    for sample in samples:
        transform_a = point_cloud_registration(body_a.markers, sample.markers_a)
        transform_b = point_cloud_registration(body_b.markers, sample.markers_b)

        tip_in_tracker = transform_point(transform_a, body_a.tip)
        tip_in_b = transform_point(invert_transform(transform_b), tip_in_tracker)
        sample_point = transform_point(identity_registration, tip_in_b)

        surface_point, distance = closest_point_on_mesh(
            sample_point, vertices, triangles, accel=accel
        )
        distance = float(np.linalg.norm(surface_point - sample_point))
        if distance < 1e-2:
            distance = 0.0
        else:
            distance = float(np.round(distance, 3))

        results.append(
            MatchResult(
                sample_point=sample_point,
                closest_point=surface_point,
                distance=distance,
            )
        )

    return results

