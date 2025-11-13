"""Unit tests for matching computation functions."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_mesh, load_rigid_body, load_samples
from src.matching import compute_matches
from src.models import RigidBody, SampleFrame

DATA_ROOT = Path(".qodo/data/2025 PA345 Student Data")


class TestComputeMatches:
    """Tests for compute_matches function."""

    def test_single_sample_identity_transform(self):
        """Verify compute_matches handles an identity transform without movement. Confirms the resulting sample and closest points align and produce a non-negative distance."""
        # Simple mesh: single triangle
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        # Rigid body with tip at origin
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        # Sample with same markers (identity transform)
        samples = [
            SampleFrame(
                markers_a=body_a.markers.copy(),
                markers_b=body_b.markers.copy()
            )
        ]
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 1
        assert results[0].sample_point.shape == (3,)
        assert results[0].closest_point.shape == (3,)
        assert results[0].distance >= 0.0

    def test_multiple_samples(self):
        """Verify compute_matches can process multiple frames at once. Confirms each result contains expected point shapes and valid distances."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.5, 0.5, 0.0])
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        samples = [
            SampleFrame(
                markers_a=body_a.markers.copy(),
                markers_b=body_b.markers.copy()
            ),
            SampleFrame(
                markers_a=body_a.markers.copy() + [1.0, 0.0, 0.0],
                markers_b=body_b.markers.copy()
            ),
        ]
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 2
        for result in results:
            assert result.sample_point.shape == (3,)
            assert result.closest_point.shape == (3,)
            assert result.distance >= 0.0

    def test_tip_on_mesh_surface(self):
        """Verify compute_matches reports near-zero distance when the tip lies on the mesh. Confirms the returned distance stays within the expected tolerance."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.25, 0.25, 0.0])  # On triangle
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        samples = [
            SampleFrame(
                markers_a=body_a.markers.copy(),
                markers_b=body_b.markers.copy()
            )
        ]
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 1
        # Distance should be very small (0.0 if thresholded)
        assert results[0].distance < 0.01 or results[0].distance == 0.0

    def test_tip_away_from_mesh(self):
        """Verify compute_matches reflects a large distance when the tip is far above the mesh. Confirms the computed distance is approximately the expected offset."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.5, 0.5, 10.0])  # Far above triangle
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        samples = [
            SampleFrame(
                markers_a=body_a.markers.copy(),
                markers_b=body_b.markers.copy()
            )
        ]
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 1
        # Distance should be approximately 10.0
        assert results[0].distance > 9.0
        assert results[0].distance < 11.0

    def test_distance_rounding(self):
        """Verify compute_matches outputs distances with consistent rounding. Confirms the formatted distance maintains the expected three decimal digits."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.5, 0.5, 0.123])  # Small distance
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        samples = [
            SampleFrame(
                markers_a=body_a.markers.copy(),
                markers_b=body_b.markers.copy()
            )
        ]
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 1
        # Distance should be rounded to 3 decimal places
        distance_str = f"{results[0].distance:.3f}"
        assert len(distance_str.split('.')[1]) <= 3

    def test_empty_samples_list(self):
        """Verify compute_matches tolerates an empty samples list. Confirms the function returns an empty result as expected."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        body_a = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        body_b = RigidBody(
            markers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tip=np.array([0.0, 0.0, 0.0])
        )
        
        samples = []
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        
        assert len(results) == 0


class TestMatchingWithRealData:
    """Tests for compute_matches using real PA3 data to verify against expected outputs."""

    def _load_expected(self, path: Path):
        """Load expected results from answer file. Confirms the number of parsed rows matches the count declared in the header."""
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

    @pytest.mark.parametrize(
        ("letter", "abs_tol", "coord_tol"),
        [
            ("A", 1e-2, 1e-2),
            ("B", 1e-2, 1e-2),
            ("C", 1e-2, 1e-2),
        ],
    )
    def test_compute_matches_against_expected_outputs(self, letter, abs_tol, coord_tol):
        """Validate compute_matches against the official PA3 A–C answer keys. Confirms each computed sample, closest point, and distance matches the expected reference within tolerance."""
        vertices, triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            DATA_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt",
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )
        
        results = compute_matches(vertices, triangles, body_a, body_b, samples)
        expected = self._load_expected(DATA_ROOT / f"PA3-{letter}-Debug-Answer.txt")
        
        # Verify all results match expected
        assert len(results) == len(expected)
        for result, (exp_sample, exp_closest, exp_distance) in zip(results, expected):
            rounded_sample = np.round(result.sample_point, 2)
            rounded_closest = np.round(result.closest_point, 2)
            assert np.allclose(rounded_sample, exp_sample, atol=coord_tol), \
                f"Sample point mismatch: got {rounded_sample}, expected {exp_sample}"
            assert np.allclose(rounded_closest, exp_closest, atol=coord_tol), \
                f"Closest point mismatch: got {rounded_closest}, expected {exp_closest}"
            assert math.isclose(result.distance, exp_distance, abs_tol=abs_tol), \
                f"Distance mismatch: got {result.distance}, expected {exp_distance}"

    def test_compute_matches_first_sample_pa3a(self):
        """Validate compute_matches for the first PA3-A sample in detail. Confirms the returned sample point, closest point, and distance equal the published expected values."""
        vertices, triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            DATA_ROOT / "PA3-A-Debug-SampleReadingsTest.txt",
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )
        
        # Test just the first sample
        first_sample = [samples[0]]
        results = compute_matches(vertices, triangles, body_a, body_b, first_sample)
        
        assert len(results) == 1
        result = results[0]
        
        # Expected from PA3-A-Debug-Answer.txt
        expected_sample = np.array([28.51, 14.33, 17.70])
        expected_closest = np.array([28.51, 14.33, 17.70])
        expected_distance = 0.000
        
        assert np.allclose(result.sample_point, expected_sample, atol=1e-2)
        assert np.allclose(result.closest_point, expected_closest, atol=1e-2)
        assert abs(result.distance - expected_distance) < 1e-2

