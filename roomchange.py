#!/usr/bin/env python3
"""RoomChange3D: compare two room videos using COLMAP sparse SfM + 3D voxel differences."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial import cKDTree


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(list(map(str, cmd)), cwd=cwd)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}")


def extract_frames(video: Path, out_dir: Path, sample_fps: float, max_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, int(round(fps / max(sample_fps, 0.1))))
    saved, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            # Resize large phone frames while retaining detail sufficient for SfM.
            h, w = frame.shape[:2]
            scale = min(1.0, 1600.0 / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            path = out_dir / f"frame_{saved:05d}.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"Failed writing {path}")
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    if saved < 5:
        raise RuntimeError(f"Only extracted {saved} frames from {video}; record a longer video.")
    print(f"Extracted {saved} frames from {video} (source frames={total})")
    return saved


def run_colmap(frames_dir: Path, workspace: Path):
    executable = shutil.which("colmap")
    if not executable:
        raise RuntimeError("COLMAP executable not found. Install COLMAP and add `colmap` to PATH.")
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    workspace.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    run([executable, "feature_extractor",
         "--database_path", database,
         "--image_path", frames_dir,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "SIMPLE_RADIAL",
         "--SiftExtraction.use_gpu", "0"])
    run([executable, "exhaustive_matcher",
         "--database_path", database,
         "--SiftMatching.use_gpu", "0"])
    run([executable, "mapper",
         "--database_path", database,
         "--image_path", frames_dir,
         "--output_path", sparse])
    models = sorted([p for p in sparse.iterdir() if p.is_dir() and any(p.iterdir())])
    if not models:
        raise RuntimeError(
            f"COLMAP could not reconstruct {frames_dir}. Try more overlap, slower motion, better light, or more frames."
        )
    model = models[0]
    # Export COLMAP model to PLY using model_converter.
    ply_path = workspace / "sparse.ply"
    run([executable, "model_converter",
         "--input_path", model,
         "--output_path", ply_path,
         "--output_type", "PLY"])
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if len(cloud.points) < 100:
        raise RuntimeError(f"Reconstruction is too sparse ({len(cloud.points)} points) for comparison.")
    return cloud, model


def clean_cloud(pcd: o3d.geometry.PointCloud):
    pcd = pcd.voxel_down_sample(voxel_size=max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.002)
    if len(pcd.points) > 50:
        _, inds = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(inds)
    return pcd


def align_clouds(before, after, voxel_size, use_ransac=True):
    """Return after transformed into before coordinates."""
    a = clean_cloud(before)
    b = clean_cloud(after)
    scale = max(np.linalg.norm(a.get_max_bound() - a.get_min_bound()), 1e-6)
    # Coarse downsample and FPFH feature matching gives a robust initial pose for different viewpoints.
    coarse = max(voxel_size * 2.5, scale / 80.0)
    a_down = a.voxel_down_sample(coarse)
    b_down = b.voxel_down_sample(coarse)
    a_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=coarse * 2.5, max_nn=40))
    b_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=coarse * 2.5, max_nn=40))
    f_a = o3d.pipelines.registration.compute_fpfh_feature(
        a_down, o3d.geometry.KDTreeSearchParamHybrid(radius=coarse * 5, max_nn=100))
    f_b = o3d.pipelines.registration.compute_fpfh_feature(
        b_down, o3d.geometry.KDTreeSearchParamHybrid(radius=coarse * 5, max_nn=100))
    threshold = coarse * 2.0
    if use_ransac and len(a_down.points) >= 20 and len(b_down.points) >= 20:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            b_down, a_down, f_b, f_a, True, threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        init = result.transformation
    else:
        init = np.eye(4)
    # Multi-scale ICP refines the estimate.
    trans = init
    for size in [coarse, max(coarse / 2, voxel_size), voxel_size]:
        a_level = a.voxel_down_sample(size)
        b_level = b.voxel_down_sample(size)
        a_level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=size * 2.5, max_nn=40))
        b_level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=size * 2.5, max_nn=40))
        reg = o3d.pipelines.registration.registration_icp(
            b_level, a_level, size * 2.5, trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        trans = reg.transformation
        print(f"ICP fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.4f}, voxel={size:.4f}")
    aligned = o3d.geometry.PointCloud(after)
    aligned.transform(trans)
    return aligned, trans


def voxel_counts(points, voxel_size):
    coords = np.floor(np.asarray(points) / voxel_size).astype(np.int64)
    groups = defaultdict(list)
    for p, c in zip(points, coords):
        groups[tuple(c)].append(p)
    return groups


def region_components(voxels):
    """6-connected components of changed voxel coordinates."""
    remaining = set(voxels)
    comps = []
    offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    while remaining:
        seed = remaining.pop()
        stack, comp = [seed], [seed]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in offsets:
                nb = (x+dx, y+dy, z+dz)
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)
    return comps


def summarize_changes(before, after, voxel_size, min_voxels=3):
    bg = voxel_counts(np.asarray(before.points), voxel_size)
    ag = voxel_counts(np.asarray(after.points), voxel_size)
    # A voxel is "observed" if it has at least one sparse SfM point. Sparse occupancy is noisy,
    # so use component size filtering and label as observation difference, not proof of object change.
    after_only = set(ag) - set(bg)
    before_only = set(bg) - set(ag)
    results = []
    change_points = []
    for kind, voxels in [("new_or_more_observed", after_only),
                         ("missing_or_less_observed", before_only)]:
        for comp in region_components(voxels):
            if len(comp) < min_voxels:
                continue
            arr = np.asarray(comp, dtype=float) * voxel_size + voxel_size / 2
            center = arr.mean(axis=0)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            results.append({
                "type": kind,
                "voxel_count": len(comp),
                "center_xyz": [round(float(v), 4) for v in center],
                "bounds_min_xyz": [round(float(v), 4) for v in lo],
                "bounds_max_xyz": [round(float(v), 4) for v in hi],
                "approx_extent": [round(float(v), 4) for v in (hi - lo + voxel_size)],
                "interpretation": (
                    "Possible added/moved object surface OR newly observed surface; check visibility."
                    if kind == "new_or_more_observed" else
                    "Possible removed/moved object surface OR surface hidden/not reconstructed; check visibility."
                )
            })
            change_points.extend(arr.tolist())
    results.sort(key=lambda x: x["voxel_count"], reverse=True)
    return results, np.asarray(change_points, dtype=float).reshape((-1, 3))


def make_html(before, after, changed, output_path):
    traces = []
    def add_cloud(cloud, name, color, size, opacity):
        pts = np.asarray(cloud.points)
        if len(pts) > 25000:
            pts = pts[np.random.default_rng(1).choice(len(pts), 25000, replace=False)]
        traces.append(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers", name=name,
            marker=dict(size=size, color=color, opacity=opacity)))
    add_cloud(before, "Before reconstruction", "#4776C5", 2, 0.65)
    add_cloud(after, "After aligned reconstruction", "#36A879", 2, 0.55)
    if len(changed):
        traces.append(go.Scatter3d(x=changed[:,0], y=changed[:,1], z=changed[:,2],
            mode="markers", name="Changed voxel centers",
            marker=dict(size=4, color="#E64B35", opacity=0.95,
                        line=dict(width=0.2, color="#FFFFFF"))))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="RoomChange3D — aligned scene and candidate change regions",
        scene=dict(xaxis_title="X (reconstruction units)", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=45, b=0))
    fig.write_html(str(output_path), include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True, type=Path, help="Baseline room video")
    ap.add_argument("--after", required=True, type=Path, help="Later room video")
    ap.add_argument("--out", default=Path("results"), type=Path)
    ap.add_argument("--fps", default=2.0, type=float, help="Frames sampled per second")
    ap.add_argument("--max-frames", default=80, type=int)
    ap.add_argument("--voxel-size", default=0.08, type=float)
    ap.add_argument("--min-voxels", default=3, type=int)
    args = ap.parse_args()
    if not args.before.exists() or not args.after.exists():
        ap.error("Both --before and --after files must exist.")
    if args.voxel_size <= 0:
        ap.error("--voxel-size must be positive.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    before_frames = out / "frames_before"
    after_frames = out / "frames_after"
    extract_frames(args.before, before_frames, args.fps, args.max_frames)
    extract_frames(args.after, after_frames, args.fps, args.max_frames)
    before_cloud, before_model = run_colmap(before_frames, out / "colmap_before")
    after_cloud, after_model = run_colmap(after_frames, out / "colmap_after")
    before_cloud = clean_cloud(before_cloud)
    after_aligned, transform = align_clouds(before_cloud, after_cloud, args.voxel_size)
    changes, changed_points = summarize_changes(before_cloud, after_aligned, args.voxel_size, args.min_voxels)
    o3d.io.write_point_cloud(str(out / "before.ply"), before_cloud)
    o3d.io.write_point_cloud(str(out / "after_aligned.ply"), after_aligned)
    change_cloud = o3d.geometry.PointCloud()
    if len(changed_points):
        change_cloud.points = o3d.utility.Vector3dVector(changed_points)
        change_cloud.paint_uniform_color([0.9, 0.18, 0.1])
    o3d.io.write_point_cloud(str(out / "changes.ply"), change_cloud)
    make_html(before_cloud, after_aligned, changed_points, out / "changes.html")
    payload = {
        "system": "RoomChange3D baseline",
        "voxel_size_reconstruction_units": args.voxel_size,
        "before_points": len(before_cloud.points),
        "after_points_aligned": len(after_aligned.points),
        "alignment_transform_after_to_before": transform.tolist(),
        "change_region_count": len(changes),
        "changes": changes,
        "caveat": "Regions are differences in sparse reconstructed observations. Occlusion, viewpoint, lighting, and SfM density can mimic additions/removals. Review the 3D view before treating a region as a real object change."
    }
    (out / "changes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RoomChange3D report", "",
        f"- Before sparse points: **{len(before_cloud.points)}**",
        f"- After sparse points (aligned): **{len(after_aligned.points)}**",
        f"- Candidate regions: **{len(changes)}**",
        f"- Voxel size: `{args.voxel_size}` reconstruction units", "",
        "## Candidate changes", ""
    ]
    if not changes:
        lines.append("No connected changed regions passed the minimum-size filter. This does not prove the room is unchanged.")
    for i, c in enumerate(changes, 1):
        lines += [
            f"### Region {i}: `{c['type']}`",
            f"- Voxel count: {c['voxel_count']}",
            f"- Center (x, y, z): `{c['center_xyz']}`",
            f"- Approximate extent: `{c['approx_extent']}`",
            f"- Interpretation: {c['interpretation']}", ""
        ]
    lines += [
        "## Confidence and limitations", "",
        "These are geometry-based candidate regions, not definitive semantic object detections. A region can be caused by an added/removed/moved object, occlusion, viewpoint coverage, lighting, or reconstruction noise. Inspect `changes.html` and verify whether the changed region is visible in both recordings.", ""
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nDone. Open:", out / "changes.html")
    print("Report:", out / "report.md")
    print("JSON:", out / "changes.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
