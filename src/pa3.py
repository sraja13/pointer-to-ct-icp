#!/usr/bin/env python3
"""
PA3 matching phase for a simplified ICP pipeline.

This script parses the provided mesh, rigid-body definition, and sample readings
files to compute the pointer tip position for each sample frame and locate the
closest point on the mesh surface. Results are written in the format prescribed
for PA3 output files.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class RigidBody:
    markers: np.ndarray  # shape (N, 3)
    tip: np.ndarray  # shape (3,)


@dataclass
class SampleFrame:
    markers_a: np.ndarray  # shape (NA, 3)
    markers_b: np.ndarray  # shape (NB, 3)


@dataclass
class MatchResult:
    sample_point: np.ndarray
    closest_point: np.ndarray
    distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PA3 closest-point matches for pointer samples."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to mesh file (Problem3Mesh.sur).",
    )
    parser.add_argument(
        "--body-a",
        type=Path,
        required=True,
        help="Path to rigid body definition for body A (Problem3-BodyA.txt).",
    )
    parser.add_argument(
        "--body-b",
        type=Path,
        required=True,
        help="Path to rigid body definition for body B (Problem3-BodyB.txt).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to sample readings file (pa3-*-SampleReadings*.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for PA3 output file (pa3-X-Output.txt).",
    )
    return parser.parse_args()


def _parse_floats_from_line(line: str, expected_count: int | None = None) -> np.ndarray:
    tokens = [tok for tok in line.replace(",", " ").split() if tok]
    if expected_count is not None and len(tokens) < expected_count:
        raise ValueError(f"Expected at least {expected_count} values, found {tokens}")
    return np.array([float(tok) for tok in tokens[:expected_count]], dtype=float)


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        num_vertices_line = fh.readline()
        if not num_vertices_line:
            raise ValueError(f"Mesh file {path} is empty.")
        num_vertices = int(num_vertices_line.strip())

        vertices = np.zeros((num_vertices, 3), dtype=float)
        for i in range(num_vertices):
            line = fh.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file while reading vertices (read {i} of {num_vertices})."
                )
            vertices[i] = _parse_floats_from_line(line, 3)

        num_triangles_line = fh.readline()
        if not num_triangles_line:
            raise ValueError("Missing triangle count in mesh file.")
        num_triangles = int(num_triangles_line.strip())

        triangles = np.zeros((num_triangles, 3), dtype=np.int32)
        for i in range(num_triangles):
            line = fh.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file while reading triangles (read {i} of {num_triangles})."
                )
            tokens = [int(tok) for tok in line.split()]
            if len(tokens) < 3:
                raise ValueError(f"Triangle record must contain 3 vertex indices: {line}")
            triangles[i] = np.array(tokens[:3], dtype=np.int32)

    return vertices, triangles


def load_rigid_body(path: Path) -> RigidBody:
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise ValueError(f"Rigid body file {path} is empty.")
        header_tokens = header.split()
        if not header_tokens:
            raise ValueError(f"Invalid header in rigid body file {path!s}: {header!r}")
        num_markers = int(header_tokens[0])

        markers = np.zeros((num_markers, 3), dtype=float)
        for i in range(num_markers):
            line = fh.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file while reading markers from {path} "
                    f"(read {i} of {num_markers})."
                )
            markers[i] = _parse_floats_from_line(line, 3)

        tip_line = fh.readline()
        if not tip_line:
            raise ValueError(f"Missing tip specification in {path}.")
        tip = _parse_floats_from_line(tip_line, 3)

    return RigidBody(markers=markers, tip=tip)


def _parse_sample_header(line: str) -> Tuple[int, int, str]:
    parts = [part.strip() for part in line.strip().split(",", maxsplit=2)]
    if len(parts) < 2:
        raise ValueError(f"Invalid sample readings header: {line!r}")
    total_markers = int(parts[0])
    num_samples = int(parts[1])
    label = parts[2] if len(parts) > 2 else ""
    return total_markers, num_samples, label


def load_samples(path: Path, num_a: int, num_b: int) -> Tuple[List[SampleFrame], str]:
    with path.open("r", encoding="utf-8") as fh:
        header_line = fh.readline()
        if not header_line:
            raise ValueError(f"Sample readings file {path} is empty.")
        total_markers, num_samples, label = _parse_sample_header(header_line)

        num_dummy = total_markers - num_a - num_b
        if num_dummy < 0:
            raise ValueError(
                f"Sample file declares {total_markers} markers, but body files account for "
                f"{num_a + num_b}. Cannot have negative dummy markers."
            )

        samples: List[SampleFrame] = []
        for sample_idx in range(num_samples):
            markers_a = np.zeros((num_a, 3), dtype=float)
            markers_b = np.zeros((num_b, 3), dtype=float)

            for i in range(num_a):
                line = fh.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF while reading A markers for sample {sample_idx}."
                    )
                markers_a[i] = _parse_floats_from_line(line, 3)

            for i in range(num_b):
                line = fh.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF while reading B markers for sample {sample_idx}."
                    )
                markers_b[i] = _parse_floats_from_line(line, 3)

            for i in range(num_dummy):
                line = fh.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF while skipping dummy markers for sample {sample_idx}."
                    )
                # Dummy markers ignored.

            samples.append(SampleFrame(markers_a=markers_a, markers_b=markers_b))

    return samples, label


def point_cloud_registration(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    # Implementation follows the SVD-based absolute orientation method also used in
    # our earlier CIS PA1 reference solution (see github.com/Rorodino/CIS-PA1).
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target must have the same shape. "
            f"Got {source_points.shape} vs {target_points.shape}."
        )
    if source_points.shape[0] < 3:
        raise ValueError("At least three non-collinear points are required for registration.")

    centroid_source = source_points.mean(axis=0)
    centroid_target = target_points.mean(axis=0)

    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    covariance = centered_source.T @ centered_target
    u, _, vt = np.linalg.svd(covariance)

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = centroid_target - rotation @ centroid_source

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation

    inv_transform = np.eye(4, dtype=float)
    inv_transform[:3, :3] = inv_rotation
    inv_transform[:3, 3] = inv_translation
    return inv_transform


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = point
    transformed = transform @ homogeneous
    return transformed[:3]


def closest_point_on_triangle(
    point: np.ndarray, tri_vertices: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    a, b, c = tri_vertices
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def closest_point_on_mesh(
    point: np.ndarray, vertices: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, float]:
    best_point = None
    best_dist_sq = math.inf
    for tri_indices in triangles:
        a = vertices[tri_indices[0]]
        b = vertices[tri_indices[1]]
        c = vertices[tri_indices[2]]
        candidate = closest_point_on_triangle(point, (a, b, c))
        dist_sq = np.sum((candidate - point) ** 2)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = candidate
    if best_point is None:
        raise RuntimeError("No triangles were processed when searching for closest point.")
    return best_point, math.sqrt(best_dist_sq)


def compute_matches(
    vertices: np.ndarray,
    triangles: np.ndarray,
    body_a: RigidBody,
    body_b: RigidBody,
    samples: List[SampleFrame],
) -> List[MatchResult]:
    identity_registration = np.eye(4, dtype=float)

    results: List[MatchResult] = []
    for sample in samples:
        transform_a = point_cloud_registration(body_a.markers, sample.markers_a)
        transform_b = point_cloud_registration(body_b.markers, sample.markers_b)

        tip_in_tracker = transform_point(transform_a, body_a.tip)
        tip_in_b = transform_point(invert_transform(transform_b), tip_in_tracker)
        sample_point = transform_point(identity_registration, tip_in_b)

        surface_point, distance = closest_point_on_mesh(sample_point, vertices, triangles)
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


def format_output_line(sample_point: np.ndarray, closest_point: np.ndarray, distance: float) -> str:
    return (
        f"{sample_point[0]:9.2f} {sample_point[1]:9.2f} {sample_point[2]:9.2f}"
        f"        {closest_point[0]:9.2f} {closest_point[1]:9.2f} {closest_point[2]:9.2f}"
        f"     {distance:6.3f}"
    )


def main() -> None:
    args = parse_args()

    vertices, triangles = load_mesh(args.mesh)
    body_a = load_rigid_body(args.body_a)
    body_b = load_rigid_body(args.body_b)
    samples, _sample_label = load_samples(
        args.samples, num_a=body_a.markers.shape[0], num_b=body_b.markers.shape[0]
    )

    results = compute_matches(vertices, triangles, body_a, body_b, samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        header_label = args.output.name
        fh.write(f"{len(results)} {header_label} 0\n")
        for result in results:
            fh.write(
                format_output_line(result.sample_point, result.closest_point, result.distance)
            )
            fh.write("\n")


if __name__ == "__main__":
    main()

