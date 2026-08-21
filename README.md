# Local 3D Semantic Voxel Map

完整的中文实现说明见
[`LOCAL_3D_VOXEL_COST_PIPELINE.md`](LOCAL_3D_VOXEL_COST_PIPELINE.md)，其中记录了
从 `/grids_points`、初始位姿参考、滚动三维 voxel 融合、通行代价计算到全局/局部
代价云发布，以及测试终点生成器和 FAR 联调的完整数据链路。

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
- `semantic_lable`: canonical integer class ID used by recorded `/grids_points`;
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

### Supported input profiles

The same executable supports both existing data sources. Select one launch
profile per node instance so their topics and timestamp domains are never mixed:

| Profile | Launch file | Input schema | Label handling | Tracking frame |
|---|---|---|---|---|
| Simulation | `semantic_voxel_map.launch` | flattened `semantic_color`, no traversability field | packed RGB is remapped to canonical IDs; terrain height cost is inferred | `world` |
| Recorded bag/live grid | `grids_points_voxel_map.launch` | unorganized `semantic_lable` plus `traversability` | IDs 0-18 and measured cost pass through | `map` |

The two profiles load independent mappings because their on-wire colors do not
mean the same thing. The simulation profile maps packed colors such as
`0x000000 -> 0` (terrain), `0x808080 -> 2` (structure), `0x0000ff -> 3`
(low obstacle), and `0xff00ff -> 11` (dynamic); the park-path omni profile also
maps `0x98fb98 -> 9` for grass/stair terrain. The bag profile performs no packed
color mapping: its integer `semantic_lable` IDs 0-18 are loaded directly with
the Cityscapes-style class table in `grids_points_voxel_map.yaml`. Each profile's
class palette independently controls `voxel_cloud.rgb`.

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

For the simulation profile, which has no traversability field, the published
snapshot additionally builds a terrain height neighborhood from labels 0, 1 and
9. Missing-cost terrain columns within 0.20 m are compared; if their height
difference is greater than 0.15 m plus a small numerical epsilon, voxels on both
sides are raised to cost 1.0. The simulation map uses 0.10 m XY cells and 0.05 m
Z cells, preventing a 0.10 m stair rise from expanding to 0.20 m solely through
center-of-voxel quantization. This identifies ground/stair boundaries while
leaving flat terrain low-cost.
Voxels that have ever received measured traversability are excluded from this
geometry inference, so `/grids_points` costs are never overwritten.

## Run and display

```bash
# Packed-RGB simulation cloud
roslaunch local3d_semantic_voxel_map semantic_voxel_map.launch rviz:=true

# Recorded/live /grids_points cloud
roslaunch local3d_semantic_voxel_map grids_points_voxel_map.launch rviz:=true
```

Published topics:

- `~semantic_voxels`: voxel-sized RViz `CUBE_LIST`, colored by semantic class;
- `~traversability_voxels`: green-yellow-red cost cubes;
- `~voxel_cloud`: `PointCloud2` with `rgb`, `label`, `semantic_confidence`,
  `traversability`, `intensity`, and `observations` fields.
- `~global_semantic_admission_grid`: a filtered view of the same local voxel
  snapshot (simulation: 0.10 m XY and 0.05 m Z; bag: isotropic 0.10 m), not a
  persistent global map. It contains `x/y/z` (`float32`),
  `traversability` (`float32`), and the upstream-compatible `semantic_lable`
  (`uint32`). Coordinates are frozen in the configured robot frame at the input
  acquisition time. Every non-dynamic voxel is retained, including ordinary
  low-cost terrain and static objects. Dynamic labels 11-18 remain in
  `~voxel_cloud` but are excluded here by default. A semantic-ground stair edge
  raised by the height rule is still present, but carries high cost while its
  local label stays 0/1/9.
- `/semantic_pcl/global_admitted`: the same filtered snapshot converted directly
  to SSMI's `x/y/z/rgb/semantic_color` schema. Simulation and bag labels use one
  canonical output encoding even though their input palettes differ. Both color
  fields contain identical packed bits. A point whose final traversability is
  at least `~ssmi_obstacle_traversability_threshold` (default 0.75) is encoded as
  a standard obstacle; other known labels retain their canonical semantic
  encoding. SSMI can therefore distinguish obstacle semantics from traversable
  terrain without a separate conversion node.
- `~confirmed` mirrors the selected local snapshot; `~rejected_dynamic` contains
  excluded labels 11-18. `~revocation_candidates` shows remembered SSMI
  obstacle voxels currently receiving positive free evidence, and
  `~revoked_free` emits the exact voxels whose multi-frame check completed.
  `~rejected_unknown`, `~rejected_rear`, and `~revoked_reclassified` remain
  empty in the current policy.

Insertion remains an immediate 0.10 m local snapshot with no multi-frame
stability gate, 0.40 m re-voxelization, or rear corridor. A separate 0.10 m
mirror remembers only obstacle-like endpoints already sent to SSMI. Mere
absence, rolling-window pruning, and dynamic occlusion never revoke an endpoint.
Five distinct frames over at least 0.5 s of confident low-cost terrain or
configured raw-depth ray traversal are required. Simulation enables ray
evidence to remove trails from movers mislabeled as static; the processed
`/grids_points` profile disables ray evidence and accepts only direct semantic/
traversability contradiction.

All map clouds, including the direct SSMI cloud, use the last successfully
processed input acquisition stamp.
The publish timer repeats that acquisition-time snapshot and never refreshes
its header when the active input stops. This node owns the initial
`global_frame -> map_start` static transform (`map -> map_start` for the bag
profile); do not launch another publisher for that child frame.

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

## Height-admission diagnostics

`diagnose_height_admission.py` checks the simulation height rule through all
three exact-stamp snapshots: `~voxel_cloud`,
`~global_semantic_admission_grid`, and `/semantic_pcl/global_admitted`. Run it
after the voxel node:

```bash
rosrun local3d_semantic_voxel_map diagnose_height_admission.py \
  _csv_path:=/tmp/height_admission_diagnostics.csv
```

The CSV records the terrain and height-obstacle counts, the maximum triggering
height difference and point pair, the robot pose obtained from TF at the cloud
acquisition stamp, and whether every admitted high-cost terrain point received
SSMI's obstacle encoding. It also audits `/semantic_pcl/global_admitted`
directly: label-9 staircase points are separated into retained green points,
high-cost points recolored to the canonical wall color, and unexpected colors.
The CSV reports both color fields, expected-encoding mismatches, and the spatial
extent of each recolored staircase patch. It also publishes RViz clouds in the
diagnostic node's private namespace:

- `~height_obstacle_voxels`: all terrain voxels raised to obstacle cost;
- `~within_column_voxels`: triggers caused within one XY column;
- `~neighbor_column_voxels`: triggers caused by neighboring columns;
- `~downstream_height_obstacles`: the selected points in the admission frame;
- `~downstream_encoding_mismatches`: points not encoded as SSMI obstacles.
- `~ssmi_stair_green`: staircase points that remain canonical terrain green;
- `~ssmi_stair_recolored_wall`: staircase points recolored by the high-cost
  SSMI obstacle rule;
- `~ssmi_stair_unexpected_color`: staircase points with neither expected color.

All diagnostic clouds retain the source acquisition stamp. A missing point,
occlusion, or timer republish is not interpreted as height evidence.

`voxel_size_xy` and `voxel_size_z` configure independent planar and vertical
resolution. The legacy `voxel_size` parameter remains supported and supplies
both defaults when either axis-specific parameter is absent.

## Offline voxel-to-FAR planning test

The supplied integration launch plays the recorded `/grids_points`, builds the
voxel map, selects a local `/way_point`, and runs FAR `localPlanner` to publish
`/path`. It deliberately does not start `pathFollower`, so the test cannot send
velocity commands to the robot.

Source both workspaces with extension enabled so ROS can find this package and
the external `local_planner` package:

```bash
source /opt/ros/noetic/setup.bash
source /home/yanaibo/mapless_navigation/far_planner/devel/setup.bash
source /home/yanaibo/mapless_navigation/local_3d_semantic_voxel/local3DSemanticVoxelMap/build/devel/setup.bash --extend
roslaunch local3d_semantic_voxel_map grids_points_far_planner_test.launch
```

This recording uses `map` as its global frame and `wuba_base` as its vehicle
frame. The integration launch passes those frame names consistently to the
voxel map, goal selector, FAR planner, and RViz. At the first valid cloud, the
voxel node records `map <- wuba_base(t0)` and publishes it as the static
`map -> map_start` transform. Voxel storage and RViz use `map_start`, so the
robot's first cloud pose is `(0, 0, 0)` with zero relative yaw. The global cost
cloud is transformed back to `map` for FAR, while the local cost cloud remains
in `wuba_base`; existing odometry and planning interfaces therefore stay
consistent.

To use live topics without playing the bundled bag, run:

```bash
roslaunch local3d_semantic_voxel_map grids_points_far_planner_test.launch \
  play_bag:=false rviz:=true
```

Useful test outputs are `/local_3d_semantic_voxel_map/voxel_cloud`,
`/local_3d_semantic_voxel_map/traversability_cost_cloud`, `/way_point`, `/path`,
and `/free_paths`.

The integration launch also starts `record_path_goal_alignment.py` by default.
For every `/path`, it transforms the active `/way_point` from `map` into the
path's frame (`wuba_base`) at exactly `path.header.stamp`. It writes a compact
summary to `/tmp/path_goal_alignment.csv` and every full path point sequence to
`/tmp/path_goal_alignment_paths.jsonl`. In RViz, `Path Goal Alignment` shows the
transformed goal in green, the selected path endpoint in red, and their error as
a yellow line. The axes display is explicitly attached to `wuba_base`.

The output file and reach tolerance can be changed at launch time:

```bash
roslaunch local3d_semantic_voxel_map grids_points_far_planner_test.launch \
  path_goal_csv:=/tmp/my_path_test.csv path_goal_tolerance:=0.5
```

Set `record_path_goal:=false` to disable the recorder. FAR publishes a finite
local motion primitive rather than a path that is guaranteed to terminate at
the waypoint, so use `min_path_error_xy`, `endpoint_error_xy`, and `goal_age_s`
in the CSV to distinguish primitive geometry from a stale waypoint.

The test launch enables FAR's local-goal watchdog. Three consecutive
`/test_local_goal_selector/valid=false` messages, or no fresh valid goal for
`0.5 s` in the odometry/header timestamp domain, makes `/path` a one-point stop
path. A later valid goal automatically resumes replanning. Override these with
`goal_timeout` and `invalid_goal_count_threshold`; setting `goal_timeout:=0`
disables only the timestamp timeout.

For `/grids_points`, the rolling voxel storage region is an oriented cuboid in
the incoming cloud frame (`wuba_base`). Configure it with
`local_box_min_{x,y,z}` and `local_box_max_{x,y,z}`. The box follows the current
robot position and orientation, and points outside it are rejected before
fusion; old voxels outside the moved box are pruned at the publish rate.
`local_radius` remains available as a spherical fallback when
`local_box_enabled=false`. `max_range` is an independent spherical sensor
outlier filter; values `<=0` disable it. The bundled cuboid configuration
disables `max_range` because the box already provides finite input bounds.

For simulation, `robot_body_exclusion_enabled` removes an asymmetric planar
rectangle in the incoming `base_link` cloud before coordinate transformation,
voxel fusion, and terrain-height inference. The bundled simulation profiles use
`x=[-0.5, 0.3] m` and `y=[-0.3, 0.3] m` for all heights, with the larger
rearward extent accounting for the forward camera placement. This prevents leg
and body returns from creating false height obstacles. The recorded
`/grids_points` profile explicitly disables this source-specific filter.
