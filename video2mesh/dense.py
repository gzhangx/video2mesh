"""Dense point clouds: COLMAP patch-match when CUDA exists, otherwise CPU stereo."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from video2mesh.colmap_io import Camera, FramePose, Reconstruction, camera_matrix, read_model
from video2mesh.colmap_options import STEREO_INPUT_TYPE, ColmapCli, option
from video2mesh.deps import command_argv, require_colmap, run_command
from video2mesh.errors import ReconstructionError

logger = logging.getLogger("video2mesh")


def build_point_cloud(
    *,
    sparse: Reconstruction,
    image_dir: Path,
    workspace: Path,
    preset: dict,
    use_cuda: bool,
    dense: bool,
    colmap_exe: str | None,
    undistorted_model: Path | None,
    backend: str,
) -> tuple[o3d.geometry.PointCloud, str]:
    """Return a colored point cloud and the name of the method that produced it.

    The sparse cloud is always available. When ``dense`` is true and COLMAP has
    CUDA, patch-match stereo plus fusion replaces it. Otherwise consecutive
    undistorted frames are matched with OpenCV's block matcher and fused with
    the sparse points.
    """
    sparse_cloud = sparse_point_cloud(sparse)
    if not dense:
        logger.info("Dense reconstruction disabled; using %d sparse points", len(sparse.points))
        return sparse_cloud, "sparse"

    if use_cuda and backend == "cli":
        fused = _try_patch_match(workspace, colmap_exe)
        if fused is not None and len(fused.points) > 0:
            logger.info("CUDA patch-match cloud: %d points", len(fused.points))
            return fused, "colmap_patch_match"
        logger.warning("Patch-match stereo did not produce a cloud. Falling back to CPU stereo.")

    if not use_cuda:
        logger.info(
            "COLMAP patch-match stereo needs CUDA, which is not available. "
            "Building a dense cloud on the CPU with OpenCV stereo and the sparse points."
        )

    model = sparse
    images = image_dir
    if undistorted_model is not None and undistorted_model.exists():
        try:
            model = read_model(undistorted_model)
            candidate = workspace / "dense" / "images"
            if candidate.is_dir() and any(candidate.iterdir()):
                images = candidate
            logger.info("CPU stereo is using undistorted pinhole images in %s", images)
        except ReconstructionError as exc:
            logger.warning("Could not read the undistorted model (%s). Using original frames.", exc)

    stereo = stereo_point_cloud(model, images, preset)
    merged = _merge_clouds(sparse_cloud, stereo)
    if len(merged.points) < 30:
        raise ReconstructionError(
            "The point cloud has fewer than 30 points after dense fusion. "
            "The sparse model may be too thin, or the frames may be too blurry."
        )
    method = "opencv_stereo" if stereo is not None and len(stereo.points) > 0 else "sparse"
    logger.info("Point cloud ready: %d points via %s", len(merged.points), method)
    return merged, method


def sparse_point_cloud(reconstruction: Reconstruction) -> o3d.geometry.PointCloud:
    if not reconstruction.points:
        raise ReconstructionError("The sparse model has no 3D points.")
    points = np.stack([point.xyz for point in reconstruction.points]).astype(np.float64)
    colors = np.stack([point.rgb for point in reconstruction.points]).astype(np.float64) / 255.0
    return _cloud_from_arrays(points, colors)


def stereo_point_cloud(
    reconstruction: Reconstruction,
    image_dir: Path,
    preset: dict,
) -> o3d.geometry.PointCloud | None:
    """Semi-dense cloud from sequential rectified pairs."""
    views = _views(reconstruction, image_dir)
    if len(views) < 2:
        logger.warning("Need at least two views for CPU stereo.")
        return None

    sparse_xyz = (
        np.stack([point.xyz for point in reconstruction.points])
        if reconstruction.points
        else None
    )
    num_disp = int(preset.get("stereo_num_disparities", 128))
    stride = max(1, int(preset.get("stereo_stride", 3)))
    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    pairs = _pair_indexes(len(views))
    logger.info("CPU stereo on %d image pairs", len(pairs))
    for left_index, right_index in pairs:
        try:
            xyz, rgb = _match_pair(views[left_index], views[right_index], num_disp, stride)
        except cv2.error as exc:
            logger.warning(
                "Stereo failed for %s / %s: %s",
                views[left_index].pose.name,
                views[right_index].pose.name,
                exc,
            )
            continue
        if xyz is None or len(xyz) == 0:
            continue
        if sparse_xyz is not None:
            xyz, rgb = _keep_near_sparse(xyz, rgb, sparse_xyz)
        if len(xyz) == 0:
            continue
        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)
        logger.debug(
            "Pair %s-%s contributed %d points",
            views[left_index].pose.name,
            views[right_index].pose.name,
            len(xyz),
        )

    if not chunks_xyz:
        logger.warning("CPU stereo produced no points. The mesh will use the sparse cloud only.")
        return None
    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    if len(xyz) > 350_000:
        choice = np.linspace(0, len(xyz) - 1, 350_000).astype(np.int64)
        xyz = xyz[choice]
        rgb = rgb[choice]
    cloud = _cloud_from_arrays(xyz, rgb)
    cloud = _cleanup_cloud(cloud, sparse_xyz)
    logger.info("CPU stereo cloud: %d points", len(cloud.points))
    return cloud


class _View:
    def __init__(self, pose: FramePose, camera: Camera, path: Path):
        self.pose = pose
        self.camera = camera
        self.path = path
        self.matrix, self.dist = camera_matrix(camera)


def _views(reconstruction: Reconstruction, image_dir: Path) -> list[_View]:
    views: list[_View] = []
    for pose in reconstruction.ordered_frames():
        camera = reconstruction.cameras.get(pose.camera_id)
        if camera is None:
            continue
        path = image_dir / pose.name
        if not path.is_file():
            logger.warning("Missing image for %s", pose.name)
            continue
        views.append(_View(pose, camera, path))
    return views


def _pair_indexes(count: int) -> list[tuple[int, int]]:
    """Consecutive pairs, plus a stride-3 pair when the clip is long enough."""
    pairs: list[tuple[int, int]] = []
    for index in range(count - 1):
        pairs.append((index, index + 1))
    if count >= 6:
        for index in range(0, count - 3, 3):
            pairs.append((index, index + 3))
    return pairs


def _match_pair(
    left: _View,
    right: _View,
    num_disparities: int,
    stride: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    rotation = right.pose.rotation @ left.pose.rotation.T
    translation = right.pose.tvec - rotation @ left.pose.tvec
    baseline = float(np.linalg.norm(translation))
    if baseline < 1e-8:
        return None, None

    image_left = cv2.imread(str(left.path), cv2.IMREAD_COLOR)
    image_right = cv2.imread(str(right.path), cv2.IMREAD_COLOR)
    if image_left is None or image_right is None:
        return None, None
    height, width = image_left.shape[:2]
    if image_right.shape[:2] != (height, width):
        image_right = cv2.resize(image_right, (width, height), interpolation=cv2.INTER_AREA)

    size = (width, height)
    rect_r1, rect_r2, proj1, _proj2, q_matrix, _roi1, _roi2 = cv2.stereoRectify(
        left.matrix,
        left.dist,
        right.matrix,
        right.dist,
        size,
        rotation,
        translation.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.25,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        left.matrix, left.dist, rect_r1, proj1, size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        right.matrix, right.dist, rect_r2, _proj2, size, cv2.CV_32FC1
    )
    rect_left = cv2.remap(image_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(image_right, map2x, map2y, cv2.INTER_LINEAR)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    ndisp = max(16, int(num_disparities) // 16 * 16)
    ndisp = min(ndisp, max(16, (width // 16) * 16))
    min_disp = -ndisp // 2
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=ndisp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=2,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (np.abs(disparity) >= 1.0)
    if int(valid.sum()) < 50:
        return None, None

    rectified_points = cv2.reprojectImageTo3D(disparity, q_matrix)
    ys, xs = np.nonzero(valid)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]
    if len(xs) > 40_000:
        take = np.linspace(0, len(xs) - 1, 40_000).astype(np.int64)
        ys = ys[take]
        xs = xs[take]
    pts_rect = rectified_points[ys, xs]
    # X_rect = R_rect * X_cam, so X_cam = R_rect^T * X_rect.
    pts_cam = (rect_r1.T @ pts_rect.T).T
    in_front = np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 1e-4)
    pts_cam = pts_cam[in_front]
    if len(pts_cam) == 0:
        return None, None
    # Drop the far tail of this pair; it is usually sky or a mismatch.
    depth_cap = float(np.percentile(pts_cam[:, 2], 95)) * 1.2
    keep = pts_cam[:, 2] < depth_cap
    pts_cam = pts_cam[keep]
    if len(pts_cam) == 0:
        return None, None
    colors = rect_left[ys, xs][in_front][keep][:, ::-1].astype(np.float64) / 255.0
    world = (left.pose.rotation.T @ (pts_cam - left.pose.tvec).T).T
    return world, colors


def _keep_near_sparse(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sparse_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = sparse_xyz.min(axis=0)
    high = sparse_xyz.max(axis=0)
    extent = np.maximum(high - low, 1e-6)
    margin = np.maximum(0.25 * extent, 1e-3)
    mask = np.all((xyz >= low - margin) & (xyz <= high + margin), axis=1)
    return xyz[mask], rgb[mask]


def _merge_clouds(
    sparse_cloud: o3d.geometry.PointCloud,
    stereo: o3d.geometry.PointCloud | None,
) -> o3d.geometry.PointCloud:
    if stereo is None or len(stereo.points) == 0:
        return _cleanup_cloud(sparse_cloud, np.asarray(sparse_cloud.points))
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(sparse_cloud.points), np.asarray(stereo.points)])
    )
    if sparse_cloud.has_colors() and stereo.has_colors():
        merged.colors = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(sparse_cloud.colors), np.asarray(stereo.colors)])
        )
    return _cleanup_cloud(merged, np.asarray(sparse_cloud.points))


def _cleanup_cloud(
    cloud: o3d.geometry.PointCloud,
    reference_xyz: np.ndarray | None,
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return cloud
    finite = np.isfinite(points).all(axis=1)
    cloud = cloud.select_by_index(np.flatnonzero(finite).tolist())
    if len(cloud.points) == 0:
        return cloud
    diag = _diagonal(reference_xyz if reference_xyz is not None and len(reference_xyz) else np.asarray(cloud.points))
    voxel = max(diag / 350.0, 1e-6)
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) >= 40:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cloud


def _cloud_from_arrays(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64), 0.0, 1.0))
    return cloud


def _diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(max(np.linalg.norm(extent), 1e-6))


def _try_patch_match(workspace: Path, colmap_exe: str | None) -> o3d.geometry.PointCloud | None:
    dense_dir = workspace / "dense"
    if not dense_dir.is_dir():
        logger.warning("No undistorted dense workspace; cannot run patch-match stereo.")
        return None
    exe = require_colmap(colmap_exe)
    fused_path = dense_dir / "fused.ply"
    try:
        cli = ColmapCli(exe)
        logger.info("COLMAP patch_match_stereo")
        patch_argv = ["patch_match_stereo"]
        cli.add(
            patch_argv,
            "patch_match_stereo",
            [
                option("workspace_path", "workspace_path", dense_dir),
                option("workspace_format", "workspace_format", "COLMAP"),
                # Numeric 1, not the word "true": some COLMAP builds parse "true" as 0.
                option("geom_consistency", "PatchMatchStereo.geom_consistency", "1"),
            ],
        )
        run_command(command_argv(exe, patch_argv), echo="info")
        logger.info("COLMAP stereo_fusion")
        fuse_argv = ["stereo_fusion"]
        cli.add(
            fuse_argv,
            "stereo_fusion",
            [
                option("workspace_path", "workspace_path", dense_dir),
                option("workspace_format", "workspace_format", "COLMAP"),
                option("input_type", STEREO_INPUT_TYPE, "geometric"),
                option("output_path", "output_path", fused_path),
            ],
        )
        run_command(command_argv(exe, fuse_argv), echo="info")
    except ReconstructionError as exc:
        logger.warning("Patch-match stereo failed:\n%s", exc)
        return None
    if not fused_path.is_file():
        return None
    cloud = o3d.io.read_point_cloud(str(fused_path))
    if len(cloud.points) == 0:
        return None
    return _cleanup_cloud(cloud, np.asarray(cloud.points))
