# RoomChange3D — compare two room recordings

A practical research prototype for detecting **likely 3D scene changes** between two phone videos of the same indoor area. It builds sparse 3D reconstructions with COLMAP, aligns them with Open3D, compares occupied space, and writes a 3D HTML visualization plus a JSON/text report.

> **Important limitation:** monocular phone videos do not provide reliable absolute scale by themselves. This prototype aligns the reconstructions geometrically and reports likely changed regions, not guaranteed object-level truth. Occlusion, lighting changes, textureless walls, and large viewpoint differences can create false positives. The report explicitly calls out ambiguous regions.

## Pipeline

1. Extract overlapping frames from each recording.
2. Run COLMAP Structure-from-Motion for each recording.
3. Convert each sparse reconstruction to a point cloud.
4. Align recording B to recording A with robust feature-based initialization + ICP.
5. Voxelize the aligned clouds and find regions observed in one recording but not the other.
6. Export `changes.html` (interactive 3D point clouds and change voxels), `changes.json`, and a readable `report.md`.

This version prioritizes a reproducible baseline over pretending that monocular geometry can always distinguish a moved object from an occluded one. A future extension can add instance segmentation and a learned multi-view model.

## Requirements

- Python 3.10+
- [COLMAP](https://colmap.github.io/) installed and available as `colmap` on PATH
- Python dependencies listed in `requirements.txt`
- Two short videos, ideally 10–30 seconds each, covering the same room with substantial overlap.

Install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

Install COLMAP separately using its official instructions. Verify:

```bash
colmap -h
```

## Record the videos well

- Walk slowly around the room, keeping walls and fixed features visible.
- Record from several viewpoints with 50–70% overlap.
- Keep the same room lighting where possible.
- Avoid motion blur and fast pans.
- Record the baseline video first, then move/add/remove only a few objects, then record again.
- Try to include some unchanged walls/furniture in both videos so alignment has stable anchors.
- For stronger scale consistency, include a known-size object or place two visible markers a measured distance apart.

## Run

```bash
python roomchange.py \
  --before data/before.mp4 \
  --after data/after.mp4 \
  --out results \
  --fps 2 \
  --max-frames 80 \
  --voxel-size 0.08
```

`--voxel-size` is in reconstruction units (not guaranteed meters). Try `0.04` for finer detail or `0.12` for noisier footage.

Outputs:

- `results/before.ply`, `results/after_aligned.ply`: sparse point clouds
- `results/changes.ply`: point cloud showing changed voxel centers
- `results/changes.html`: interactive 3D view
- `results/changes.json`: machine-readable regions
- `results/report.md`: human-readable summary
- `results/frames_before/`, `results/frames_after/`: sampled frames
- `results/colmap_before/`, `results/colmap_after/`: reconstruction workspaces

## Reading the output

Each changed region is categorized as:
- `new_or_more_observed`: voxels with points in the after reconstruction but not the before reconstruction
- `missing_or_less_observed`: voxels with points in the before reconstruction but not the after reconstruction

Those are **observation differences**, not automatic proof that an object was added or removed. If a region is visible from one camera path and hidden from the other, it can appear as a change. Review the HTML scene and report together.

## Troubleshooting

- **COLMAP finds too few images / sparse cloud:** record more slowly, improve lighting, avoid blank walls, increase `--max-frames`, and ensure the videos overlap.
- **Alignment is poor:** increase unchanged overlap, lower `--voxel-size`, or try `--no-ransac` only if the initial alignment is already close.
- **Many changes on walls:** likely viewpoint/coverage or reconstruction differences. Record from more similar paths and use larger voxels.
- **No changes detected:** try a larger voxel size or move objects more clearly.

## Design choices and tradeoffs

- **COLMAP** supplies established SfM camera-pose estimation and sparse triangulation without training a custom model.
- **Open3D** handles point-cloud alignment, outlier removal, and visualization.
- **Voxel occupancy** is intentionally simple and explainable; it is less sensitive to exact point-to-point distances but depends on reconstruction density.
- The system favors conservative, inspectable evidence over overconfident natural-language claims. Semantic object labels are not yet automatic in this baseline; the report describes spatial regions and recommends visual review. Add a detector/segmenter only after geometry is stable.
