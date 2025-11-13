"""Unit tests for I/O functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.io import load_mesh, load_rigid_body, load_samples
from src.models import RigidBody, SampleFrame


class TestLoadMesh:
    """Tests for load_mesh function."""

    def test_load_simple_mesh(self):
        """Verify load_mesh parses a minimal .sur file. Confirms the returned vertices and triangles equal the expected arrays."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            f.write("3\n")  # 3 vertices
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 0.0 0.0\n")
            f.write("0.0 1.0 0.0\n")
            f.write("1\n")  # 1 triangle
            f.write("0 1 2\n")
            temp_path = Path(f.name)

        try:
            vertices, triangles = load_mesh(temp_path)
            
            assert vertices.shape == (3, 3)
            assert triangles.shape == (1, 3)
            assert np.allclose(vertices[0], [0.0, 0.0, 0.0])
            assert np.allclose(vertices[1], [1.0, 0.0, 0.0])
            assert np.allclose(vertices[2], [0.0, 1.0, 0.0])
            assert np.array_equal(triangles[0], [0, 1, 2])
        finally:
            temp_path.unlink()

    def test_load_mesh_with_comma_separated_values(self):
        """Verify load_mesh accepts comma-separated vertex coordinates. Confirms the parsed vertices match the expected floating-point values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            f.write("2\n")
            f.write("1.0, 2.0, 3.0\n")
            f.write("4.0, 5.0, 6.0\n")
            f.write("0\n")
            temp_path = Path(f.name)

        try:
            vertices, triangles = load_mesh(temp_path)
            
            assert vertices.shape == (2, 3)
            assert triangles.shape == (0, 3)
            assert np.allclose(vertices[0], [1.0, 2.0, 3.0])
            assert np.allclose(vertices[1], [4.0, 5.0, 6.0])
        finally:
            temp_path.unlink()

    def test_empty_file_raises_error(self):
        """Ensure load_mesh rejects an empty file. Confirms a ValueError is raised as expected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="empty"):
                load_mesh(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_triangle_count_raises_error(self):
        """Ensure load_mesh fails when the triangle count line is absent. Confirms the raised ValueError message matches the expected text."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            f.write("1\n")
            f.write("0.0 0.0 0.0\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing triangle count"):
                load_mesh(temp_path)
        finally:
            temp_path.unlink()

    def test_incomplete_vertices_raises_error(self):
        """Ensure load_mesh detects incomplete vertex data. Confirms the resulting ValueError reports the unexpected end-of-file condition."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            f.write("2\n")
            f.write("0.0 0.0 0.0\n")
            # Missing second vertex
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unexpected end of file"):
                load_mesh(temp_path)
        finally:
            temp_path.unlink()

    def test_invalid_triangle_indices_raises_error(self):
        """Ensure load_mesh validates triangle index length. Confirms a ValueError is thrown when fewer than three indices are provided."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sur', delete=False) as f:
            f.write("3\n")
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 0.0 0.0\n")
            f.write("0.0 1.0 0.0\n")
            f.write("1\n")
            f.write("0 1\n")  # Only 2 indices instead of 3
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must contain 3 vertex indices"):
                load_mesh(temp_path)
        finally:
            temp_path.unlink()


class TestLoadRigidBody:
    """Tests for load_rigid_body function."""

    def test_load_simple_rigid_body(self):
        """Verify load_rigid_body parses a minimal rigid body definition. Confirms the resulting RigidBody markers and tip match the authored values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("2 Problem3-BodyA.txt\n")  # 2 markers
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            f.write("7.0 8.0 9.0\n")  # Tip
            temp_path = Path(f.name)

        try:
            body = load_rigid_body(temp_path)
            
            assert isinstance(body, RigidBody)
            assert body.markers.shape == (2, 3)
            assert body.tip.shape == (3,)
            assert np.allclose(body.markers[0], [1.0, 2.0, 3.0])
            assert np.allclose(body.markers[1], [4.0, 5.0, 6.0])
            assert np.allclose(body.tip, [7.0, 8.0, 9.0])
        finally:
            temp_path.unlink()

    def test_load_rigid_body_with_comma_separated_values(self):
        """Verify load_rigid_body tolerates comma-separated input. Confirms the markers and tip align with the expected coordinates."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1\n")
            f.write("1.0, 2.0, 3.0\n")
            f.write("4.0, 5.0, 6.0\n")
            temp_path = Path(f.name)

        try:
            body = load_rigid_body(temp_path)
            
            assert body.markers.shape == (1, 3)
            assert np.allclose(body.markers[0], [1.0, 2.0, 3.0])
            assert np.allclose(body.tip, [4.0, 5.0, 6.0])
        finally:
            temp_path.unlink()

    def test_empty_file_raises_error(self):
        """Ensure load_rigid_body rejects an empty file. Confirms a ValueError is raised as expected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="empty"):
                load_rigid_body(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_tip_raises_error(self):
        """Ensure load_rigid_body requires a tip definition. Confirms the expected ValueError is emitted when the tip line is absent."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1\n")
            f.write("1.0 2.0 3.0\n")
            # Missing tip
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing tip"):
                load_rigid_body(temp_path)
        finally:
            temp_path.unlink()

    def test_incomplete_markers_raises_error(self):
        """Ensure load_rigid_body detects missing marker entries. Confirms a ValueError signaling unexpected EOF is raised."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("2\n")
            f.write("1.0 2.0 3.0\n")
            # Missing second marker
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unexpected end of file"):
                load_rigid_body(temp_path)
        finally:
            temp_path.unlink()


class TestLoadSamples:
    """Tests for load_samples function."""

    def test_load_simple_samples(self):
        """Verify load_samples parses a basic readings file with labels. Confirms the returned SampleFrame objects and label match the expected structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("4, 2, TestLabel\n")  # 4 total markers, 2 samples
            # Sample 0: 2 markers A, 2 markers B
            f.write("1.0 2.0 3.0\n")  # A marker 0
            f.write("4.0 5.0 6.0\n")  # A marker 1
            f.write("7.0 8.0 9.0\n")  # B marker 0
            f.write("10.0 11.0 12.0\n")  # B marker 1
            # Sample 1: 2 markers A, 2 markers B
            f.write("13.0 14.0 15.0\n")  # A marker 0
            f.write("16.0 17.0 18.0\n")  # A marker 1
            f.write("19.0 20.0 21.0\n")  # B marker 0
            f.write("22.0 23.0 24.0\n")  # B marker 1
            temp_path = Path(f.name)

        try:
            samples, label = load_samples(temp_path, num_a=2, num_b=2)
            
            assert label == "TestLabel"
            assert len(samples) == 2
            assert isinstance(samples[0], SampleFrame)
            assert samples[0].markers_a.shape == (2, 3)
            assert samples[0].markers_b.shape == (2, 3)
            assert np.allclose(samples[0].markers_a[0], [1.0, 2.0, 3.0])
            assert np.allclose(samples[1].markers_a[0], [13.0, 14.0, 15.0])
        finally:
            temp_path.unlink()

    def test_load_samples_with_dummy_markers(self):
        """Verify load_samples skips dummy markers correctly. Confirms the resulting SampleFrame shapes align with the expected counts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("5, 1\n")  # 5 total markers, 1 sample, no label
            f.write("1.0 2.0 3.0\n")  # A marker 0
            f.write("4.0 5.0 6.0\n")  # A marker 1
            f.write("7.0 8.0 9.0\n")  # B marker 0
            f.write("10.0 11.0 12.0\n")  # Dummy marker 0 (ignored)
            f.write("13.0 14.0 15.0\n")  # Dummy marker 1 (ignored) - need 2 dummy markers (5 total - 2 A - 1 B = 2)
            temp_path = Path(f.name)

        try:
            samples, label = load_samples(temp_path, num_a=2, num_b=1)
            
            assert len(samples) == 1
            assert samples[0].markers_a.shape == (2, 3)
            assert samples[0].markers_b.shape == (1, 3)
        finally:
            temp_path.unlink()

    def test_empty_file_raises_error(self):
        """Ensure load_samples rejects an empty file. Confirms the raised ValueError communicates the empty input condition."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="empty"):
                load_samples(temp_path, num_a=1, num_b=1)
        finally:
            temp_path.unlink()

    def test_negative_dummy_markers_raises_error(self):
        """Ensure load_samples detects inconsistent marker counts. Confirms a ValueError is raised when dummy markers would be negative."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("2, 1\n")  # 2 total markers
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            temp_path = Path(f.name)

        try:
            # num_a=2, num_b=2 means 4 markers needed, but file says 2
            with pytest.raises(ValueError, match="negative dummy markers"):
                load_samples(temp_path, num_a=2, num_b=2)
        finally:
            temp_path.unlink()

    def test_incomplete_sample_raises_error(self):
        """Ensure load_samples flags incomplete frame data. Confirms the ValueError references the unexpected EOF."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("4, 1\n")
            f.write("1.0 2.0 3.0\n")  # A marker 0
            # Missing remaining markers
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unexpected EOF"):
                load_samples(temp_path, num_a=2, num_b=2)
        finally:
            temp_path.unlink()

    def test_invalid_header_raises_error(self):
        """Ensure load_samples validates the header format. Confirms a ValueError is raised when the header cannot be parsed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid sample readings header"):
                load_samples(temp_path, num_a=1, num_b=1)
        finally:
            temp_path.unlink()

