"""Output formatting functions for PA3 results."""

import numpy as np


def format_output_line(sample_point: np.ndarray, closest_point: np.ndarray, distance: float) -> str:
    """
    Format a single result line for PA3 output.
    
    Args:
        sample_point: Sample point coordinates, shape (3,)
        closest_point: Closest point on mesh, shape (3,)
        distance: Distance between points
    
    Returns:
        Formatted string for output file
    """
    return (
        f"{sample_point[0]:9.2f} {sample_point[1]:9.2f} {sample_point[2]:9.2f}"
        f"        {closest_point[0]:9.2f} {closest_point[1]:9.2f} {closest_point[2]:9.2f}"
        f"     {distance:6.3f}"
    )

