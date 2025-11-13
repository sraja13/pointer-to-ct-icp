import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pa3 import (  # noqa: E402
    MatchResult,
    compute_matches,
    load_mesh,
    load_rigid_body,
    load_samples,
)


DATA_ROOT = Path("data/2025 PA345 Student Data")


def _load_expected(path: Path):
    """Load an answer file into structured tuples. Confirms the number of parsed rows equals the header-specified count."""
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        expected_count = int(header[0])
        rows = []
        for line in fh:
            tokens = [float(tok) for tok in line.split()]
            sample = np.array(tokens[0:3])
            closest = np.array(tokens[3:6])
            distance = tokens[6]
            rows.append((sample, closest, distance))
    assert len(rows) == expected_count
    return rows


def _compare_results(
    results: list[MatchResult],
    expected_rows,
    abs_tol: float = 1e-2,
    coord_tol: float = 1e-2,
):
    """Compare computed MatchResult objects to expected tuples. Confirms sample points, closest points, and distances align within the given tolerances."""
    assert len(results) == len(expected_rows)
    for result, (exp_sample, exp_closest, exp_distance) in zip(results, expected_rows):
        rounded_sample = np.round(result.sample_point, 2)
        rounded_closest = np.round(result.closest_point, 2)
        assert np.allclose(rounded_sample, exp_sample, atol=coord_tol)
        assert np.allclose(rounded_closest, exp_closest, atol=coord_tol)
        assert math.isclose(result.distance, exp_distance, abs_tol=abs_tol)


def _run_dataset(letter: str):
    """Execute the full PA3 pipeline for a dataset letter. Confirms the function returns both the computed results and the loaded expected rows."""
    mesh_vertices, mesh_triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
    body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
    body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
    samples, _ = load_samples(
        DATA_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt",
        num_a=body_a.markers.shape[0],
        num_b=body_b.markers.shape[0],
    )
    results = compute_matches(mesh_vertices, mesh_triangles, body_a, body_b, samples)
    expected = _load_expected(DATA_ROOT / f"PA3-{letter}-Debug-Answer.txt")
    return results, expected


@pytest.mark.parametrize(
    ("letter", "abs_tol", "coord_tol"),
    [
        ("A", 1e-2, 1e-2),
        ("B", 1e-2, 1e-2),
        ("C", 1e-2, 1e-2),
        ("D", 1e-2, 2e-2),
        ("E", 2e-2, 2e-2),
        ("F", 2e-2, 2e-2),
    ],
)
def test_pa3_debug_datasets(letter: str, abs_tol: float, coord_tol: float):
    """Validate the modular PA3 pipeline against debug datasets A–F. Confirms each computed match agrees with the official answers within the prescribed tolerances."""
    results, expected = _run_dataset(letter)
    _compare_results(results, expected, abs_tol=abs_tol, coord_tol=coord_tol)

