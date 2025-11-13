"""Geometry functions for computing closest points on triangles and meshes."""

import math
from typing import Tuple

import numpy as np


def closest_point_on_triangle(
    point: np.ndarray, tri_vertices: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Find the closest point on a triangle to a given point.
    
    Uses barycentric coordinates and edge/vertex checks to find the closest
    point, which may be on a vertex, edge, or interior of the triangle.
    
    Args:
        point: Query point, shape (3,)
        tri_vertices: Tuple of three triangle vertices, each shape (3,)
    
    Returns:
        Closest point on the triangle, shape (3,)
    """
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
    """
    Find the closest point on a mesh surface to a given point.
    
    Iterates through all triangles in the mesh and finds the closest point
    on any triangle, returning the overall closest point and distance.
    
    Args:
        point: Query point, shape (3,)
        vertices: Mesh vertices, shape (N, 3)
        triangles: Triangle vertex indices, shape (M, 3)
    
    Returns:
        Tuple of (closest_point, distance) where:
        - closest_point: Closest point on mesh, shape (3,)
        - distance: Euclidean distance to closest point
    """
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

