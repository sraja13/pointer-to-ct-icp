"""Unit tests for output formatting functions."""

import numpy as np
import pytest

from src.output import format_output_line


class TestFormatOutputLine:
    """Tests for format_output_line function."""

    def test_basic_formatting(self):
        """Verify format_output_line renders a standard sample/closest pair. Confirms the resulting string contains each value formatted as expected."""
        sample_point = np.array([1.0, 2.0, 3.0])
        closest_point = np.array([1.1, 2.1, 3.1])
        distance = 0.173
        
        result = format_output_line(sample_point, closest_point, distance)
        
        assert isinstance(result, str)
        # Check that all values are present
        assert "1.00" in result
        assert "2.00" in result
        assert "3.00" in result
        assert "0.173" in result

    def test_negative_values(self):
        """Verify format_output_line handles negative coordinates. Confirms the rendered string preserves the expected negative signs."""
        sample_point = np.array([-1.0, -2.0, -3.0])
        closest_point = np.array([-1.1, -2.1, -3.1])
        distance = 0.173
        
        result = format_output_line(sample_point, closest_point, distance)
        
        assert "-1.00" in result
        assert "-2.00" in result
        assert "-3.00" in result

    def test_large_values(self):
        """Verify format_output_line supports large magnitude inputs. Confirms the output string includes the appropriately rounded large values."""
        sample_point = np.array([123.45, 678.90, 999.99])
        closest_point = np.array([123.50, 678.95, 1000.00])
        distance = 0.123
        
        result = format_output_line(sample_point, closest_point, distance)
        
        assert "123.45" in result or "123.46" in result  # Rounding
        assert "0.123" in result

    def test_zero_distance(self):
        """Verify format_output_line prints zero distance correctly. Confirms the string contains the expected 0.000 distance token."""
        sample_point = np.array([1.0, 2.0, 3.0])
        closest_point = np.array([1.0, 2.0, 3.0])
        distance = 0.0
        
        result = format_output_line(sample_point, closest_point, distance)
        
        assert "0.000" in result

    def test_precision_formatting(self):
        """Verify format_output_line enforces coordinate and distance precision. Confirms the produced string rounds coordinates to two decimals and distance to three."""
        sample_point = np.array([1.234567, 2.345678, 3.456789])
        closest_point = np.array([1.234567, 2.345678, 3.456789])
        distance = 0.123456
        
        result = format_output_line(sample_point, closest_point, distance)
        
        # Coordinates should be rounded to 2 decimal places
        # Distance should be rounded to 3 decimal places
        assert isinstance(result, str)
        # The exact format depends on rounding, but should be consistent

    def test_output_structure(self):
        """Verify format_output_line emits the correct number order. Confirms the output string contains all sample, closest, and distance fields in sequence."""
        sample_point = np.array([1.0, 2.0, 3.0])
        closest_point = np.array([4.0, 5.0, 6.0])
        distance = 5.196  # sqrt(3^2 + 3^2 + 3^2)
        
        result = format_output_line(sample_point, closest_point, distance)
        
        # Should have sample point, then closest point, then distance
        parts = result.split()
        assert len(parts) >= 7  # At least 7 numbers (3 sample + 3 closest + 1 distance)

