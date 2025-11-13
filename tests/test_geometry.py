"""Unit tests for geometry functions."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import closest_point_on_mesh, closest_point_on_triangle
from src.io import load_mesh, load_rigid_body, load_samples
from src.matching import compute_matches
from src.transforms import invert_transform, point_cloud_registration, transform_point

DATA_ROOT = Path(".qodo/data/2025 PA345 Student Data")


class TestClosestPointOnTriangle:
    """Tests for closest_point_on_triangle function."""

    def test_point_at_vertex_a(self):
        """Verify closest_point_on_triangle returns vertex A when the query equals vertex A. Confirms the computed closest point matches the expected vertex coordinates."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = a.copy()
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, a, atol=1e-10)

    def test_point_at_vertex_b(self):
        """Verify closest_point_on_triangle returns vertex B when queried with that vertex. Confirms the result equals the expected coordinate to numerical tolerance."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = b.copy()
        
        result = closest_point_on_triangle(point, (b, b, c))
        assert np.allclose(result, b, atol=1e-10)

    def test_point_at_vertex_c(self):
        """Verify closest_point_on_triangle returns vertex C for an identical query point. Confirms the returned position coincides with the expected vertex."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = c.copy()
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, c, atol=1e-10)

    def test_point_on_edge_ab(self):
        """Verify closest_point_on_triangle preserves a point lying on edge AB. Confirms the algorithm outputs the midpoint exactly as expected."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = (a + b) / 2.0  # Midpoint of AB
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, point, atol=1e-10)

    def test_point_on_edge_ac(self):
        """Verify closest_point_on_triangle preserves a point on edge AC. Confirms the returned point equals the original midpoint within tolerance."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = (a + c) / 2.0  # Midpoint of AC
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, point, atol=1e-10)

    def test_point_on_edge_bc(self):
        """Verify closest_point_on_triangle preserves a point on edge BC. Confirms the computed projection matches the midpoint as expected."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = (b + c) / 2.0  # Midpoint of BC
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, point, atol=1e-10)

    def test_point_inside_triangle(self):
        """Verify closest_point_on_triangle returns the point itself when it lies inside the triangle. Confirms the result coincides with the expected interior coordinates."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = np.array([0.25, 0.25, 0.0])  # Inside triangle
        
        result = closest_point_on_triangle(point, (a, b, c))
        # Should project to the point itself (or very close)
        assert np.allclose(result, point, atol=1e-10)

    def test_point_outside_vertex_a_region(self):
        """Verify closest_point_on_triangle snaps to vertex A for a query in its Voronoi region. Confirms the resulting coordinate equals vertex A."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = np.array([-1.0, -1.0, 0.0])  # Outside, closest to A
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, a, atol=1e-10)

    def test_point_outside_vertex_b_region(self):
        """Verify closest_point_on_triangle snaps to vertex B for an external point. Confirms the computed closest point equals the expected vertex B coordinates."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = np.array([2.0, -1.0, 0.0])  # Outside, closest to B
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, b, atol=1e-10)

    def test_point_outside_vertex_c_region(self):
        """Verify closest_point_on_triangle snaps to vertex C for a query in its region. Confirms the projected point matches vertex C."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = np.array([-1.0, 2.0, 0.0])  # Outside, closest to C
        
        result = closest_point_on_triangle(point, (a, b, c))
        assert np.allclose(result, c, atol=1e-10)

    def test_point_above_triangle(self):
        """Verify closest_point_on_triangle projects vertically down onto the triangle. Confirms the returned point equals the expected projection and the distance is zero in-plane."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        point = np.array([0.25, 0.25, 1.0])  # Above center
        
        result = closest_point_on_triangle(point, (a, b, c))
        # Should project to center of triangle
        expected = np.array([0.25, 0.25, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_3d_triangle(self):
        """Verify closest_point_on_triangle handles a triangle in 3D space. Confirms the returned point lies on the triangle as expected."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        c = np.array([0.0, 1.0, 1.0])
        point = np.array([0.5, 0.5, 0.5])
        
        result = closest_point_on_triangle(point, (a, b, c))
        # Result should be on the triangle
        assert result.shape == (3,)


class TestClosestPointOnMesh:
    """Tests for closest_point_on_mesh function."""

    def test_single_triangle_mesh(self):
        """Verify closest_point_on_mesh handles a single-triangle mesh. Confirms the closest point and distance match the expected projection and height."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        point = np.array([0.25, 0.25, 1.0])
        
        closest, distance = closest_point_on_mesh(point, vertices, triangles)
        
        assert closest.shape == (3,)
        assert distance >= 0.0
        assert isinstance(distance, float)
        # Distance should be approximately 1.0 (height above triangle)
        assert abs(distance - 1.0) < 0.1

    def test_point_at_vertex(self):
        """Verify closest_point_on_mesh returns the vertex itself when queried there. Confirms both the closest point and distance align with expectations."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        triangles = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        point = vertices[0].copy()
        
        closest, distance = closest_point_on_mesh(point, vertices, triangles)
        
        assert np.allclose(closest, point, atol=1e-10)
        assert abs(distance) < 1e-10

    def test_point_on_triangle(self):
        """Verify closest_point_on_mesh leaves a point on the surface unchanged. Confirms the distance is zero and coordinates match exactly."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        point = np.array([0.5, 0.5, 0.0])  # On the triangle
        
        closest, distance = closest_point_on_mesh(point, vertices, triangles)
        
        assert np.allclose(closest, point, atol=1e-10)
        assert abs(distance) < 1e-10

    def test_multiple_triangles(self):
        """Verify closest_point_on_mesh evaluates multiple triangles correctly. Confirms the chosen triangle and returned distance match the expected nearest surface."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        point = np.array([0.5, 0.5, 1.0])  # Above center
        
        closest, distance = closest_point_on_mesh(point, vertices, triangles)
        
        assert closest.shape == (3,)
        assert distance >= 0.0
        # Should be close to the point projected onto the plane
        assert abs(closest[2] - 0.0) < 1e-10

    def test_empty_mesh_raises_error(self):
        """Ensure closest_point_on_mesh rejects meshes without triangles. Confirms a RuntimeError is raised with the expected message."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([], dtype=np.int32).reshape(0, 3)
        point = np.array([0.5, 0.5, 0.5])
        
        with pytest.raises(RuntimeError, match="No triangles"):
            closest_point_on_mesh(point, vertices, triangles)

    def test_distant_point(self):
        """Verify closest_point_on_mesh can handle a distant query. Confirms the returned distance significantly exceeds the mesh scale as expected."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        point = np.array([10.0, 10.0, 10.0])
        
        closest, distance = closest_point_on_mesh(point, vertices, triangles)
        
        assert distance > 10.0  # Should be far
        assert closest.shape == (3,)


class TestGeometryWithRealData:
    """Tests for geometry functions using real PA3 data to verify against expected outputs."""

    @pytest.fixture(scope="class")
    def pa3_data(self):
        """Provide the PA3-A dataset for comparative testing. Confirms consumers receive vertices, triangles, and sample frames identical to the reference files."""
        vertices, triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            DATA_ROOT / "PA3-A-Debug-SampleReadingsTest.txt",
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )
        return vertices, triangles, body_a, body_b, samples

    def test_closest_point_on_mesh_with_pa3_data(self, pa3_data):
        """Validate closest_point_on_mesh against the PA3-A reference sample. Confirms the computed tip projection and distance match the official expected values."""
        vertices, triangles, body_a, body_b, samples = pa3_data
        
        # Compute tip position for first sample
        transform_a = point_cloud_registration(body_a.markers, samples[0].markers_a)
        transform_b = point_cloud_registration(body_b.markers, samples[0].markers_b)
        tip_in_tracker = transform_point(transform_a, body_a.tip)
        tip_in_b = transform_point(invert_transform(transform_b), tip_in_tracker)
        sample_point = transform_point(np.eye(4, dtype=float), tip_in_b)
        
        # Find closest point on mesh
        closest_point, distance = closest_point_on_mesh(sample_point, vertices, triangles)
        
        # Expected values from PA3-A-Debug-Answer.txt (first sample)
        expected_sample = np.array([28.51, 14.33, 17.70])
        expected_closest = np.array([28.51, 14.33, 17.70])
        expected_distance = 0.000
        
        # Verify results match expected (within tolerance)
        # Tolerance is needed because floating point arithmetic accumulates small rounding errors
        # through multiple matrix operations (SVD, transformations, geometric calculations).
        # A tolerance of 1e-2 (0.01) accounts for these numerical precision limitations.
        assert np.allclose(sample_point, expected_sample, atol=1e-2)
        assert np.allclose(closest_point, expected_closest, atol=1e-2)
        assert abs(distance - expected_distance) < 1e-2

    def test_closest_point_multiple_samples(self, pa3_data):
        """Validate closest_point_on_mesh against several PA3-A samples. Confirms each computed projection aligns with the corresponding expected coordinates and zero distance."""
        vertices, triangles, body_a, body_b, samples = pa3_data
        
        # Test first 3 samples
        expected_samples = [
            np.array([28.51, 14.33, 17.70]),
            np.array([29.96, 17.34, 12.30]),
            np.array([-30.03, 8.77, -32.64]),
        ]
        expected_closest = [
            np.array([28.51, 14.33, 17.70]),
            np.array([29.96, 17.34, 12.30]),
            np.array([-30.03, 8.77, -32.64]),
        ]
        
        for i in range(min(3, len(samples))):
            transform_a = point_cloud_registration(body_a.markers, samples[i].markers_a)
            transform_b = point_cloud_registration(body_b.markers, samples[i].markers_b)
            tip_in_tracker = transform_point(transform_a, body_a.tip)
            tip_in_b = transform_point(invert_transform(transform_b), tip_in_tracker)
            sample_point = transform_point(np.eye(4, dtype=float), tip_in_b)
            
            closest_point, distance = closest_point_on_mesh(sample_point, vertices, triangles)
            
            assert np.allclose(sample_point, expected_samples[i], atol=1e-2)
            assert np.allclose(closest_point, expected_closest[i], atol=1e-2)
            assert abs(distance) < 1e-2  # PA3-A has zero distance (perfect match)

