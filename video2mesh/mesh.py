"""Poisson and Ball-Pivoting meshes from a colored point cloud."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from video2mesh.errors import ReconstructionError
from video2mesh.progress import call_with_heartbeat, should_log_tick

logger = logging.getLogger("video2mesh")


def orient_and_mesh(
    cloud: o3d.geometry.PointCloud,
    camera_centers: np.ndarray,
    *,
    method: str,
    poisson_depth: int,
    poisson_trim: float,
) -> dict[str, o3d.geometry.TriangleMesh]:
    """Estimate normals and build the requested mesh or meshes.

    ``method`` is ``poisson``, ``bpa``, or ``both``. The primary mesh is always
    stored under the key ``mesh``. A second method is stored as ``bpa`` when
    both are requested.
    """
    logger.info("estimating normals for %d points", len(cloud.points))
    prepared = prepare_cloud(cloud, camera_centers)
    depth = _depth_for_size(poisson_depth, len(prepared.points))
    meshes: dict[str, o3d.geometry.TriangleMesh] = {}
    if method in {"poisson", "both"}:
        mesh = poisson_reconstruct(prepared, depth=depth, trim_quantile=poisson_trim)
        meshes["mesh"] = mesh
        logger.info(
            "Poisson mesh: %d vertices, %d triangles",
            len(mesh.vertices),
            len(mesh.triangles),
        )
    if method in {"bpa", "both"}:
        bpa = ball_pivot_reconstruct(prepared)
        if method == "bpa":
            meshes["mesh"] = bpa
        else:
            meshes["bpa"] = bpa
        logger.info(
            "Ball-pivoting mesh: %d vertices, %d triangles",
            len(bpa.vertices),
            len(bpa.triangles),
        )
    if "mesh" not in meshes:
        raise ReconstructionError(f"Unknown mesh method {method!r}.")
    return meshes


def prepare_cloud(
    cloud: o3d.geometry.PointCloud,
    camera_centers: np.ndarray,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) < 30:
        raise ReconstructionError(
            f"Only {len(cloud.points)} points are available. Need at least 30 to build a mesh."
        )
    points = np.asarray(cloud.points)
    extent = points.max(axis=0) - points.min(axis=0)
    diagonal = float(max(np.linalg.norm(extent), 1e-6))
    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    median = float(np.median(distances)) if len(distances) else diagonal / 50.0
    radius = max(median * 4.0, diagonal / 100.0)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    _orient_toward_cameras(cloud, camera_centers)
    return cloud


def poisson_reconstruct(
    cloud: o3d.geometry.PointCloud,
    *,
    depth: int,
    trim_quantile: float,
) -> o3d.geometry.TriangleMesh:
    logger.info("running Poisson reconstruction (depth %d)", int(depth))
    mesh, densities = call_with_heartbeat(
        "Poisson reconstruction",
        lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=int(depth),
            scale=1.1,
            linear_fit=False,
        ),
    )
    densities = np.asarray(densities)
    if trim_quantile > 0 and densities.size == len(mesh.vertices) and densities.size > 0:
        threshold = float(np.quantile(densities, trim_quantile))
        logger.info("trimming vertices below density quantile %.3f", trim_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)
    logger.info("cropping mesh to the point cloud")
    mesh = _crop_to_cloud(mesh, cloud, margin=0.08)
    logger.info("keeping the largest connected component")
    mesh = _largest_component(mesh)
    _cleanup(mesh)
    _paint_from_cloud(mesh, cloud)
    if len(mesh.triangles) == 0:
        raise ReconstructionError(
            "Poisson reconstruction produced an empty mesh. Try a lower --poisson-trim, "
            "more frames, or --mesh bpa."
        )
    mesh.compute_vertex_normals()
    return mesh


def ball_pivot_reconstruct(cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    if len(distances) == 0:
        raise ReconstructionError("Ball pivoting needs a non-empty point cloud.")
    spacing = float(np.median(distances))
    if spacing <= 0:
        spacing = 1e-3
    radii = o3d.utility.DoubleVector([spacing * 1.5, spacing * 3.0, spacing * 6.0])
    logger.info("running ball-pivoting reconstruction")
    mesh = call_with_heartbeat(
        "ball pivoting",
        lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii),
    )
    mesh = _largest_component(mesh)
    _cleanup(mesh)
    _paint_from_cloud(mesh, cloud)
    if len(mesh.triangles) == 0:
        raise ReconstructionError(
            "Ball pivoting produced an empty mesh. The cloud may be too sparse or uneven. "
            "Try --mesh poisson."
        )
    mesh.compute_vertex_normals()
    return mesh


def _orient_toward_cameras(cloud: o3d.geometry.PointCloud, camera_centers: np.ndarray) -> None:
    normals = np.asarray(cloud.normals)
    points = np.asarray(cloud.points)
    centers = np.asarray(camera_centers, dtype=np.float64)
    if len(centers) == 0 or len(points) == 0:
        return
    # Cameras are few, so a brute-force nearest center is cheaper than a tree.
    deltas = points[:, None, :] - centers[None, :, :]
    nearest = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
    toward = centers[nearest] - points
    flip = np.sum(normals * toward, axis=1) < 0
    normals[flip] *= -1
    cloud.normals = o3d.utility.Vector3dVector(normals)


def _crop_to_cloud(
    mesh: o3d.geometry.TriangleMesh,
    cloud: o3d.geometry.PointCloud,
    margin: float,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0:
        return mesh
    points = np.asarray(cloud.points)
    low = points.min(axis=0)
    high = points.max(axis=0)
    extent = np.maximum(high - low, 1e-6)
    box = o3d.geometry.AxisAlignedBoundingBox(low - margin * extent, high + margin * extent)
    return mesh.crop(box)


def _largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _areas = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh
    labels = np.asarray(labels)
    keep = labels == int(np.argmax(counts))
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def _cleanup(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()


def _paint_from_cloud(mesh: o3d.geometry.TriangleMesh, cloud: o3d.geometry.PointCloud) -> None:
    if len(mesh.vertices) == 0 or not cloud.has_colors():
        return
    if len(mesh.vertices) > 200_000:
        logger.info("Mesh is large (%d vertices); skipping color transfer", len(mesh.vertices))
        return
    tree = o3d.geometry.KDTreeFlann(cloud)
    colors = np.asarray(cloud.colors)
    vertices = np.asarray(mesh.vertices)
    total = len(vertices)
    logger.info("coloring %d mesh vertices", total)
    painted = np.empty((total, 3), dtype=np.float64)
    for index, vertex in enumerate(vertices, start=1):
        _count, indices, _dist = tree.search_knn_vector_3d(vertex, 1)
        painted[index - 1] = colors[indices[0]] if indices else (0.7, 0.7, 0.7)
        if should_log_tick(index, total):
            logger.info("colored %d/%d mesh vertices", index, total)
    mesh.vertex_colors = o3d.utility.Vector3dVector(painted)


def _depth_for_size(depth: int, npoints: int) -> int:
    if npoints < 500:
        return min(depth, 6)
    if npoints < 2000:
        return min(depth, 7)
    if npoints < 8000:
        return min(depth, 8)
    return depth
