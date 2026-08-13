# Local 3D Semantic Voxel Map

ROS1 sparse 3D voxel map. Every active voxel contains:

- the three strongest semantic hypotheses plus an `others` evidence bucket;
- normalized dominant-class confidence;
- traversability cost in `[0, 1]` (`0` easy, `1` impassable);
- observation count and last-observed time.

The semantic update follows the compact log-evidence strategy used by SSMI,
without depending on OctoMap. The storage is a bounded sparse hash map, so this
package builds on a standard ROS Noetic installation without OpenVDB.

## Input cloud

The input is `sensor_msgs/PointCloud2` with required fields `x`, `y`, `z` and:

- `semantic_color` (default): SSMI-compatible packed RGB bits in a `FLOAT32`;
- or `label`: any integer semantic class ID.

Optional fields:

- `semantic_confidence` in `[0, 1]`;
- `traversability` or `cost` in `[0, 1]`.

Categorical labels are never averaged. Points falling in the same voxel during
one scan vote by semantic confidence and produce only one map update.

SSMI flattens its 640x480 depth/semantic image into a 307200-point, one-row
cloud. This node samples fixed pixels at the configured x/y stride before any
coordinate transform. The supplied stride-3 configuration processes
213x160 = 34080 points per frame. This is image-coordinate decimation: no
semantic label is averaged or created.

`timing_report_frames` controls map-update timing reports. Each measurement
starts at entry to the point-cloud callback and ends after that frame's voxel
fusion and temporal pruning; periodic map publication is excluded. Set it to
`1` to log every frame, or keep `50` for avg/min/max windows without log spam.

`decay_seconds` is evaluated only when a successfully transformed depth frame
arrives. A voxel expires when
`current_depth_frame_stamp - voxel_last_hit_depth_frame_stamp` exceeds the
threshold. Wall time and callback processing delay are not part of this test;
if the camera stops publishing, this sensor-time decay clock also stops.

## Traversability fusion

The semantic distribution is converted to expected class cost, including the
unknown probability mass. If an input cost field exists, the target is:

```text
target = semantic_cost_weight * semantic_expected_cost
       + (1 - semantic_cost_weight) * measured_cost
```

The stored cost uses an asymmetric exponential moving average. `cost_rise_alpha`
is deliberately larger than `cost_fall_alpha`, so a dangerous observation takes
effect quickly while declaring a voxel safe again requires repeated evidence.

## Run and display

```bash
roslaunch local3d_semantic_voxel_map semantic_voxel_map.launch rviz:=true
```

Published topics:

- `~semantic_voxels`: voxel-sized RViz `CUBE_LIST`, colored by semantic class;
- `~traversability_voxels`: green-yellow-red cost cubes;
- `~voxel_cloud`: `PointCloud2` with `rgb`, `label`, `semantic_confidence`,
  `traversability`, `intensity`, and `observations` fields.

In RViz, enable one cube topic at a time when the map is large. The point cloud
can use `RGB8` for semantics or `Intensity` for traversability.

Services:

```bash
rosservice call /local_3d_semantic_voxel_map/reset
rosservice call /local_3d_semantic_voxel_map/save_map
rosservice call /local_3d_semantic_voxel_map/load_map
```

Save/load uses `~map_file` and preserves the full Top-3 semantic evidence, cost,
observation count, and timestamp of every voxel.
