"""Geometry functions for computing closest points on triangles and meshes."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class _KDNode:
    __slots__ = ("index", "axis", "left", "right")

    def __init__(self, index: int, axis: int) -> None:
        self.index = index
        self.axis = axis
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    """Minimal KD-tree for 3D points supporting k-NN and radius queries."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("KDTree expects an array of shape (N, 3)")
        indices = np.arange(len(self.points))
        self.root = self._build(indices, depth=0)
        self.size = len(self.points)

    def _build(self, indices: np.ndarray, depth: int) -> Optional[_KDNode]:
        if indices.size == 0:
            return None
        axis = depth % 3
        sorted_idx = indices[np.argsort(self.points[indices, axis])]
        median_pos = len(sorted_idx) // 2
        median_index = sorted_idx[median_pos]

        node = _KDNode(index=median_index, axis=axis)
        node.left = self._build(sorted_idx[:median_pos], depth + 1)
        node.right = self._build(sorted_idx[median_pos + 1 :], depth + 1)
        return node

    def query(self, point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError("Query point must have shape (3,)")
        if self.root is None or self.size == 0:
            return np.array([]), np.array([], dtype=int)

        k = max(1, min(int(k), self.size))
        heap: List[Tuple[float, int]] = []  # max-heap storing (-dist_sq, index)

        def search(node: Optional[_KDNode]) -> None:
            if node is None:
                return
            idx = node.index
            node_point = self.points[idx]
            dist_sq = float(np.sum((point - node_point) ** 2))

            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, idx))
            elif dist_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-dist_sq, idx))

            axis = node.axis
            diff = point[axis] - node_point[axis]
            first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            search(first)
            if len(heap) < k or diff * diff < -heap[0][0]:
                search(second)

        search(self.root)

        results = sorted((-d, idx) for d, idx in heap)
        distances = np.sqrt([d for d, _ in results])
        indices = np.array([idx for _, idx in results], dtype=int)
        return distances, indices

    def query_ball_point(self, point: np.ndarray, radius: float) -> List[int]:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError("Query point must have shape (3,)")
        if radius < 0:
            return []
        radius_sq = radius * radius
        results: List[int] = []

        def search(node: Optional[_KDNode]) -> None:
            if node is None:
                return
            idx = node.index
            node_point = self.points[idx]
            dist_sq = float(np.sum((point - node_point) ** 2))
            if dist_sq <= radius_sq:
                results.append(idx)

            axis = node.axis
            diff = point[axis] - node_point[axis]
            if diff <= 0:
                search(node.left)
                if diff * diff <= radius_sq:
                    search(node.right)
            else:
                search(node.right)
                if diff * diff <= radius_sq:
                    search(node.left)

        search(self.root)
        return results


def build_triangle_accelerator(
    vertices: np.ndarray, triangles: np.ndarray, k: int = 32
) -> Dict[str, object]:
    """
    Build a KD-tree accelerator for mesh triangle queries.

    The accelerator stores triangle centroids in a custom KD-tree along with
    per-triangle bounding radii. ``closest_point_on_mesh`` can then shortlist
    candidate triangles before running the exact projection. Passing the
    accelerator is optional; if omitted, the function falls back to a
    brute-force search.

    Args:
        vertices: Mesh vertices, shape (N, 3)
        triangles: Triangle indices, shape (M, 3)
        k: Number of nearest centroids to sample for the initial guess

    Returns:
        Dictionary containing the KD-tree and auxiliary metadata.
    """
    tri_vertices = vertices[triangles]  # (M, 3, 3)
    centroids = tri_vertices.mean(axis=1)
    radii = np.linalg.norm(tri_vertices - centroids[:, None, :], axis=2).max(axis=1)
    max_radius = float(radii.max()) if len(radii) else 0.0

    tree = KDTree(centroids) if len(centroids) else None
    return {
        "tree": tree,
        "centroids": centroids,
        "radii": radii,
        "max_radius": max_radius,
        "k": max(1, int(k)),
    }


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
    point: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    accel: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Find the closest point on a mesh surface to a given point.
    
    When an acceleration structure built via :func:`build_triangle_accelerator`
    is provided, a custom KD-tree narrows the search to nearby triangles before
    running the exact projection. If no accelerator is supplied, the function
    falls back to the original brute-force search.
    
    Args:
        point: Query point, shape (3,)
        vertices: Mesh vertices, shape (N, 3)
        triangles: Triangle vertex indices, shape (M, 3)
        accel: Optional dictionary returned by ``build_triangle_accelerator``
    
    Returns:
        Tuple of (closest_point, distance) where:
        - closest_point: Closest point on mesh, shape (3,)
        - distance: Euclidean distance to closest point
    """
    best_point = None
    best_dist_sq = math.inf

    candidate_indices = None
    considered: set[int] = set()

    if accel is not None and accel.get("tree") is not None:
        tree: KDTree = accel["tree"]  # type: ignore[assignment]
        k = min(int(accel.get("k", 32) or 32), len(triangles))
        _distances, indices = tree.query(point, k=k)
        indices = np.atleast_1d(indices).astype(int)
        candidate_indices = [idx for idx in indices if idx >= 0]
    else:
        candidate_indices = list(range(len(triangles)))

    def _evaluate(index: int, point_array: np.ndarray) -> None:
        nonlocal best_point, best_dist_sq
        tri_indices = triangles[index]
        a = vertices[tri_indices[0]]
        b = vertices[tri_indices[1]]
        c = vertices[tri_indices[2]]
        candidate = closest_point_on_triangle(point_array, (a, b, c))
        dist_sq = np.sum((candidate - point_array) ** 2)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = candidate

    for idx in candidate_indices or []:
        if idx in considered:
            continue
        considered.add(idx)
        _evaluate(idx, point)

    if accel is not None and accel.get("tree") is not None:
        centroids: np.ndarray = accel["centroids"]  # type: ignore[assignment]
        radii: np.ndarray = accel["radii"]  # type: ignore[assignment]
        max_radius: float = float(accel.get("max_radius", 0.0))

        if best_point is None:
            # No candidates processed (e.g., empty triangles): fall back to brute force loop below.
            candidate_indices = None
        else:
            best_distance = math.sqrt(best_dist_sq)
            search_radius = best_distance + max_radius
            tree: KDTree = accel["tree"]  # type: ignore[assignment]
            neighbors = tree.query_ball_point(point, search_radius)
            for idx in neighbors:
                if idx in considered:
                    continue
                # Bounding-sphere culling
                centroid = centroids[idx]
                if np.linalg.norm(centroid - point) - radii[idx] > best_distance:
                    continue
                considered.add(idx)
                _evaluate(idx, point)

    if best_point is None:
        # Fallback brute-force evaluation (handles empty accel or empty candidate list)
        for tri_idx in range(len(triangles)):
            if tri_idx in considered:
                continue
            _evaluate(tri_idx, point)

    if best_point is None:
        raise RuntimeError("No triangles were processed when searching for closest point.")
    return best_point, math.sqrt(best_dist_sq)

