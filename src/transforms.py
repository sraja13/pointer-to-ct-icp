"""Transform and registration functions for rigid body transformations."""

import numpy as np


def point_cloud_registration(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """
    Compute the rigid transformation that aligns source_points to target_points.
    
    Uses SVD-based absolute orientation method to find the optimal rotation and
    translation that minimizes the least-squares error between corresponding points.
    
    Args:
        source_points: Source point cloud, shape (N, 3)
        target_points: Target point cloud, shape (N, 3)
    
    Returns:
        4x4 homogeneous transformation matrix
    """
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
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.
    
    Args:
        transform: 4x4 homogeneous transformation matrix
    
    Returns:
        4x4 inverse transformation matrix
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation

    inv_transform = np.eye(4, dtype=float)
    inv_transform[:3, :3] = inv_rotation
    inv_transform[:3, 3] = inv_translation
    return inv_transform


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation to a 3D point.
    
    Args:
        transform: 4x4 homogeneous transformation matrix
        point: 3D point, shape (3,)
    
    Returns:
        Transformed 3D point, shape (3,)
    """
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = point
    transformed = transform @ homogeneous
    return transformed[:3]

