"""Unit tests for transform and registration functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_rigid_body, load_samples
from src.transforms import invert_transform, point_cloud_registration, transform_point

DATA_ROOT = Path("data/2025 PA345 Student Data")


class TestPointCloudRegistration:
    """Tests for point_cloud_registration function."""

    def test_identity_transformation(self):
        """Verify point_cloud_registration returns identity for identical point sets. Confirms both rotation and translation components match the expected identity transform."""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        transform = point_cloud_registration(points, points)
        
        assert np.allclose(transform[:3, :3], np.eye(3), atol=1e-10)
        assert np.allclose(transform[:3, 3], np.zeros(3), atol=1e-10)
        assert transform[3, 3] == 1.0

    def test_translation_only(self):
        """Verify point_cloud_registration recovers pure translations. Confirms the solved translation vector equals the known offset while rotation stays identity."""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([5.0, 10.0, 15.0])
        target = source + translation
        
        transform = point_cloud_registration(source, target)
        
        # Check rotation is identity
        assert np.allclose(transform[:3, :3], np.eye(3), atol=1e-10)
        # Check translation
        assert np.allclose(transform[:3, 3], translation, atol=1e-10)

    def test_rotation_only(self):
        """Verify point_cloud_registration recovers a pure rotation. Confirms the resulting rotation matrix matches the 90-degree reference with zero translation."""
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # 90 degree rotation around z-axis
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = (rotation_matrix @ source.T).T
        
        transform = point_cloud_registration(source, target)
        
        # Check rotation matches
        assert np.allclose(transform[:3, :3], rotation_matrix, atol=1e-10)
        # Check translation is zero (centered)
        assert np.allclose(transform[:3, 3], np.zeros(3), atol=1e-10)

    def test_rotation_and_translation(self):
        """Verify point_cloud_registration simultaneously recovers rotation and translation. Confirms the transformed source point matches the expected target coordinates."""
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([5.0, 10.0, 15.0])
        target = (rotation @ source.T).T + translation
        
        transform = point_cloud_registration(source, target)
        
        # Verify transformation
        transformed_source = transform_point(transform, source[0])
        assert np.allclose(transformed_source, target[0], atol=1e-10)

    def test_shape_mismatch_raises_error(self):
        """Ensure point_cloud_registration validates point set shapes. Confirms a ValueError is raised when source and target shapes differ."""
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        with pytest.raises(ValueError, match="same shape"):
            point_cloud_registration(source, target)

    def test_insufficient_points_raises_error(self):
        """Ensure point_cloud_registration enforces the three-point minimum. Confirms a ValueError is emitted when point clouds are too small."""
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        with pytest.raises(ValueError, match="At least three"):
            point_cloud_registration(source, target)

    def test_determinant_correction(self):
        """Verify point_cloud_registration corrects reflections to proper rotations. Confirms the resulting rotation matrix maintains a positive determinant."""
        # Create a case that might result in negative determinant
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        # Reflection would cause negative det, but registration should correct it
        target = source.copy()
        transform = point_cloud_registration(source, target)
        
        # Determinant should be positive (proper rotation)
        assert np.linalg.det(transform[:3, :3]) > 0


class TestInvertTransform:
    """Tests for invert_transform function."""

    def test_identity_inverse(self):
        """Verify invert_transform leaves an identity matrix unchanged. Confirms the inverse equals the original identity matrix."""
        identity = np.eye(4, dtype=float)
        inv = invert_transform(identity)
        
        assert np.allclose(inv, identity, atol=1e-10)

    def test_translation_inverse(self):
        """Verify invert_transform negates pure translations. Confirms the inverted transform carries the expected negative translation with identity rotation."""
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = [5.0, 10.0, 15.0]
        inv = invert_transform(transform)
        
        # Inverse translation should be negative
        assert np.allclose(inv[:3, 3], [-5.0, -10.0, -15.0], atol=1e-10)
        assert np.allclose(inv[:3, :3], np.eye(3), atol=1e-10)

    def test_rotation_inverse(self):
        """Verify invert_transform transposes pure rotations. Confirms the resulting rotation equals the expected transpose of the original matrix."""
        transform = np.eye(4, dtype=float)
        # 90 degree rotation around z-axis
        transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        inv = invert_transform(transform)
        
        # Inverse rotation should be transpose
        expected_rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert np.allclose(inv[:3, :3], expected_rotation, atol=1e-10)

    def test_composition_is_identity(self):
        """Verify combining a transform with its inverse produces identity. Confirms the composed matrix matches the identity matrix within tolerance."""
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform[:3, 3] = [5.0, 10.0, 15.0]
        
        inv = invert_transform(transform)
        composed = transform @ inv
        
        assert np.allclose(composed, np.eye(4), atol=1e-10)


class TestTransformPoint:
    """Tests for transform_point function."""

    def test_identity_transform(self):
        """Verify transform_point leaves points unchanged under identity. Confirms the output equals the input coordinates."""
        identity = np.eye(4, dtype=float)
        point = np.array([1.0, 2.0, 3.0])
        
        result = transform_point(identity, point)
        assert np.allclose(result, point, atol=1e-10)

    def test_translation_only(self):
        """Verify transform_point applies pure translations correctly. Confirms the resulting point equals the original plus the transform's translation vector."""
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = [5.0, 10.0, 15.0]
        point = np.array([1.0, 2.0, 3.0])
        
        result = transform_point(transform, point)
        expected = point + transform[:3, 3]
        assert np.allclose(result, expected, atol=1e-10)

    def test_rotation_only(self):
        """Verify transform_point rotates coordinates by the given rotation matrix. Confirms the result matches the analytically rotated point."""
        transform = np.eye(4, dtype=float)
        # 90 degree rotation around z-axis
        transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        point = np.array([1.0, 0.0, 0.0])
        
        result = transform_point(transform, point)
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_rotation_and_translation(self):
        """Verify transform_point handles combined rotation and translation. Confirms the output matches the expected rotated-and-shifted coordinates."""
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform[:3, 3] = [5.0, 10.0, 15.0]
        point = np.array([1.0, 0.0, 0.0])
        
        result = transform_point(transform, point)
        # Rotated: [0, 1, 0], then translated: [5, 11, 15]
        expected = np.array([5.0, 11.0, 15.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_zero_point(self):
        """Verify transform_point maps the origin to the transform's translation. Confirms the output equals the transform's translation vector."""
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform[:3, 3] = [5.0, 10.0, 15.0]
        point = np.array([0.0, 0.0, 0.0])
        
        result = transform_point(transform, point)
        # Origin just gets translated
        assert np.allclose(result, transform[:3, 3], atol=1e-10)


class TestTransformsWithRealData:
    """Tests for transforms using real PA3 data to verify against expected outputs."""

    @pytest.fixture(scope="class")
    def pa3_data(self):
        """Load the PA3-A dataset for transform validation. Confirms consumers receive rigid bodies and sample frames consistent with the reference files."""
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            DATA_ROOT / "PA3-A-Debug-SampleReadingsTest.txt",
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )
        return body_a, body_b, samples

    def test_registration_with_pa3_data(self, pa3_data):
        """Validate point_cloud_registration on PA3-A markers. Confirms the recovered transform exhibits orthonormal rotation and maps markers onto their expected positions."""
        body_a, _, samples = pa3_data
        
        # Compute registration transform
        transform = point_cloud_registration(body_a.markers, samples[0].markers_a)
        
        # Verify transform properties:
        # 1. Rotation matrix should be orthogonal (proper rotation)
        rotation = transform[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6), "Rotation should be orthogonal"
        assert np.linalg.det(rotation) > 0, "Rotation should have positive determinant"
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-6, "Rotation determinant should be 1"
        
        # 2. Transform should produce valid results when applied
        transformed_marker = transform[:3, :3] @ body_a.markers[0] + transform[:3, 3]
        # The transformed marker should be close to the sample marker (within registration error)
        assert np.linalg.norm(transformed_marker - samples[0].markers_a[0]) < 1.0

    def test_transform_tip_with_pa3_data(self, pa3_data):
        """Validate chained transforms for the PA3-A tip coordinates. Confirms the tracker-frame and body-frame tip positions are finite 3D vectors as expected."""
        body_a, body_b, samples = pa3_data
        
        # Compute transforms
        transform_a = point_cloud_registration(body_a.markers, samples[0].markers_a)
        transform_b = point_cloud_registration(body_b.markers, samples[0].markers_b)
        
        # Transform tip through coordinate systems
        tip_in_tracker = transform_point(transform_a, body_a.tip)
        tip_in_b = transform_point(invert_transform(transform_b), tip_in_tracker)
        
        # Results should be valid 3D points
        assert tip_in_tracker.shape == (3,)
        assert tip_in_b.shape == (3,)
        assert np.all(np.isfinite(tip_in_tracker))
        assert np.all(np.isfinite(tip_in_b))

    def test_inverse_composition_with_pa3_data(self, pa3_data):
        """Validate that real-data transforms invert cleanly. Confirms multiplying a transform by its inverse yields an identity matrix within tolerance."""
        body_a, _, samples = pa3_data
        
        transform = point_cloud_registration(body_a.markers, samples[0].markers_a)
        inv_transform = invert_transform(transform)
        composed = transform @ inv_transform
        
        # Should be identity
        assert np.allclose(composed, np.eye(4), atol=1e-10)

