"""File I/O functions for loading mesh, rigid bodies, and sample data."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .models import RigidBody, SampleFrame


def _parse_floats_from_line(line: str, expected_count: int | None = None) -> np.ndarray:
    """Parse floating-point values from a line, handling comma-separated values."""
    tokens = [tok for tok in line.replace(",", " ").split() if tok]
    if expected_count is not None and len(tokens) < expected_count:
        raise ValueError(f"Expected at least {expected_count} values, found {tokens}")
    return np.array([float(tok) for tok in tokens[:expected_count]], dtype=float)


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh from a .sur file.
    
    Returns:
        Tuple of (vertices, triangles) where:
        - vertices: shape (N, 3) array of vertex coordinates
        - triangles: shape (M, 3) array of triangle vertex indices
    """
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
    """
    Load a rigid body definition from a text file.
    
    Returns:
        RigidBody object with markers and tip point.
    """
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
    """Parse the header line from a sample readings file."""
    parts = [part.strip() for part in line.strip().split(",", maxsplit=2)]
    if len(parts) < 2:
        raise ValueError(f"Invalid sample readings header: {line!r}")
    total_markers = int(parts[0])
    num_samples = int(parts[1])
    label = parts[2] if len(parts) > 2 else ""
    return total_markers, num_samples, label


def load_samples(path: Path, num_a: int, num_b: int) -> Tuple[List[SampleFrame], str]:
    """
    Load sample readings from a text file.
    
    Args:
        path: Path to the sample readings file
        num_a: Number of markers for body A
        num_b: Number of markers for body B
    
    Returns:
        Tuple of (samples, label) where samples is a list of SampleFrame objects.
    """
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

