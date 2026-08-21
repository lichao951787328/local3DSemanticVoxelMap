#!/usr/bin/env python3

"""Audit terrain-height obstacles through the complete local-to-SSMI path."""

import csv
import math
import os
import struct
import threading
from collections import Counter, defaultdict

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf2_ros


TERRAIN_LABELS = frozenset((0, 1, 9))
WALL_RGB_BITS = 0x0066669C
STAIR_LABEL = 9
STAIR_RGB_BITS = 0x0098FB98

CANONICAL_RGB_BITS = (
    0x00804080, 0x00F423E8, 0x00464646, 0x0066669C, 0x00BE9999,
    0x00999999, 0x00FAAA1E, 0x00DCDC00, 0x006B8E23, 0x0098FB98,
    0x004682B4, 0x00DC143C, 0x00FF0000, 0x0000008E, 0x00000046,
    0x00003C64, 0x00005064, 0x000000E6, 0x00770B20,
)
UNKNOWN_RGB_BITS = 0x007F7F7F


def float_bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0] & 0x00FFFFFF


def packed_rgb(red, green, blue):
    bits = (red << 16) | (green << 8) | blue
    return struct.unpack("=f", struct.pack("=I", bits))[0]


DEBUG_FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("rgb", 12, PointField.FLOAT32, 1),
]


class HeightAdmissionDiagnostics:
    def __init__(self):
        self.voxel_topic = rospy.get_param(
            "~voxel_topic", "/local_3d_semantic_voxel_map/voxel_cloud")
        self.admission_topic = rospy.get_param(
            "~admission_topic",
            "/local_3d_semantic_voxel_map/global_semantic_admission_grid")
        self.ssmi_topic = rospy.get_param(
            "~ssmi_topic", "/semantic_pcl/global_admitted")
        map_namespace = rospy.get_param(
            "~map_namespace", "/local_3d_semantic_voxel_map").rstrip("/")
        legacy_voxel_size = float(rospy.get_param(
            "~voxel_size", rospy.get_param(
                map_namespace + "/voxel_size", 0.10)))
        self.voxel_size_xy = float(rospy.get_param(
            "~voxel_size_xy", rospy.get_param(
                map_namespace + "/voxel_size_xy", legacy_voxel_size)))
        self.voxel_size_z = float(rospy.get_param(
            "~voxel_size_z", rospy.get_param(
                map_namespace + "/voxel_size_z", legacy_voxel_size)))
        self.height_threshold = float(rospy.get_param(
            "~height_threshold", rospy.get_param(
                map_namespace + "/terrain_height_difference_threshold",
                0.15)))
        self.height_epsilon = float(rospy.get_param(
            "~height_epsilon", rospy.get_param(
                map_namespace + "/terrain_height_comparison_epsilon",
                1e-6)))
        self.neighborhood_radius = float(
            rospy.get_param("~neighborhood_radius", rospy.get_param(
                map_namespace + "/terrain_height_neighborhood_radius",
                0.20)))
        self.obstacle_cost = float(rospy.get_param(
            "~obstacle_cost", rospy.get_param(
                map_namespace + "/terrain_height_obstacle_cost", 1.0)))
        self.ssmi_threshold = float(rospy.get_param(
            "~ssmi_threshold", rospy.get_param(
                map_namespace + "/ssmi_obstacle_traversability_threshold",
                0.75)))
        self.log_every_n = max(1, int(rospy.get_param("~log_every_n", 5)))
        self.csv_path = rospy.get_param(
            "~csv_path", "/tmp/height_admission_diagnostics.csv")

        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.shutting_down = False
        self.messages = {
            "voxel": {},
            "admission": {},
            "ssmi": {},
        }
        self.last_processed_stamp = None
        self.frame_count = 0
        self.csv_file = None
        self.csv_writer = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._open_csv()

        self.height_pub = rospy.Publisher(
            "~height_obstacle_voxels", PointCloud2, queue_size=1, latch=True)
        self.self_reason_pub = rospy.Publisher(
            "~within_column_voxels", PointCloud2, queue_size=1, latch=True)
        self.neighbor_reason_pub = rospy.Publisher(
            "~neighbor_column_voxels", PointCloud2, queue_size=1, latch=True)
        self.downstream_pub = rospy.Publisher(
            "~downstream_height_obstacles", PointCloud2, queue_size=1, latch=True)
        self.mismatch_pub = rospy.Publisher(
            "~downstream_encoding_mismatches", PointCloud2, queue_size=1, latch=True)
        self.stair_green_pub = rospy.Publisher(
            "~ssmi_stair_green", PointCloud2, queue_size=1, latch=True)
        self.stair_wall_pub = rospy.Publisher(
            "~ssmi_stair_recolored_wall", PointCloud2, queue_size=1, latch=True)
        self.stair_other_pub = rospy.Publisher(
            "~ssmi_stair_unexpected_color", PointCloud2, queue_size=1, latch=True)

        self.subscribers = [
            rospy.Subscriber(self.voxel_topic, PointCloud2, self.callback,
                             callback_args="voxel", queue_size=3),
            rospy.Subscriber(self.admission_topic, PointCloud2, self.callback,
                             callback_args="admission", queue_size=3),
            rospy.Subscriber(self.ssmi_topic, PointCloud2, self.callback,
                             callback_args="ssmi", queue_size=3),
        ]
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "Height admission diagnostics waiting for exact-stamp snapshots: "
            "%s | %s | %s; voxel=(%.3f, %.3f, %.3f), "
            "height trigger > %.3f + %.1e m; CSV=%s",
            self.voxel_topic, self.admission_topic, self.ssmi_topic,
            self.voxel_size_xy, self.voxel_size_xy, self.voxel_size_z,
            self.height_threshold, self.height_epsilon, self.csv_path)

    def _open_csv(self):
        parent = os.path.dirname(os.path.abspath(self.csv_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        self.csv_file = open(self.csv_path, "w", newline="")
        fieldnames = [
            "stamp", "voxel_frame", "admission_frame", "voxel_points",
            "terrain_points", "height_obstacle_points", "within_column_points",
            "neighbor_column_points", "admission_points",
            "downstream_height_points", "ssmi_points", "ssmi_wall_matches",
            "ssmi_wall_mismatches", "length_mismatch", "height_min_x",
            "height_max_x", "height_min_y", "height_max_y", "height_min_z",
            "height_max_z", "maximum_within_column_dz",
            "maximum_neighbor_column_dz", "triggering_neighbor_pairs",
            "maximum_pair_ax", "maximum_pair_ay", "maximum_pair_az",
            "maximum_pair_bx", "maximum_pair_by", "maximum_pair_bz",
            "robot_map_x", "robot_map_y", "robot_map_z", "robot_roll",
            "robot_pitch", "robot_yaw", "top_base_x_bins",
            "top_base_z_bins", "ssmi_rgb_semantic_disagreements",
            "ssmi_expected_color_mismatches", "stair_points",
            "stair_high_cost_points", "stair_green_points",
            "stair_wall_points", "stair_other_color_points",
            "stair_wall_without_high_cost",
            "stair_high_cost_not_wall", "stair_wall_min_x",
            "stair_wall_max_x", "stair_wall_min_y", "stair_wall_max_y",
            "stair_wall_min_z", "stair_wall_max_z",
            "top_stair_wall_x_bins", "top_stair_wall_y_bins",
            "top_stair_wall_z_bins",
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def close(self):
        self.shutting_down = True
        with self.process_lock:
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()
                self.csv_file = None

    @staticmethod
    def stamp_key(message):
        return message.header.stamp.to_nsec()

    def callback(self, message, stream):
        key = self.stamp_key(message)
        if key == 0:
            return
        ready = None
        with self.lock:
            self.messages[stream][key] = message
            common = (set(self.messages["voxel"]) &
                      set(self.messages["admission"]) &
                      set(self.messages["ssmi"]))
            if common:
                selected = max(common)
                if selected != self.last_processed_stamp:
                    ready = tuple(self.messages[name][selected]
                                  for name in ("voxel", "admission", "ssmi"))
                    self.last_processed_stamp = selected
                for cache in self.messages.values():
                    for old_key in list(cache):
                        if old_key <= selected or len(cache) > 20:
                            del cache[old_key]
        if ready is not None:
            with self.process_lock:
                if not self.shutting_down:
                    self.process(*ready)

    def read_voxels(self, cloud):
        # sensor_msgs.point_cloud2 returns selected fields in their serialized
        # offset order. Keep this tuple in the same order as voxel_cloud.
        fields = ("x", "y", "z", "label", "measured_traversability",
                  "traversability")
        return list(point_cloud2.read_points(
            cloud, field_names=fields, skip_nans=False))

    def classify_height_reasons(self, voxels):
        columns = defaultdict(list)
        for index, (x, y, z, label, measured, _) in enumerate(voxels):
            if int(label) not in TERRAIN_LABELS or math.isfinite(measured):
                continue
            key = (int(math.floor(x / self.voxel_size_xy)),
                   int(math.floor(y / self.voxel_size_xy)))
            columns[key].append((index, z))

        within_columns = set()
        neighbor_columns = set()
        limits = {}
        maximum_within_dz = 0.0
        maximum_neighbor_dz = 0.0
        triggering_neighbor_pairs = 0
        maximum_neighbor_pair = [float("nan")] * 6
        for key, samples in columns.items():
            z_values = [sample[1] for sample in samples]
            limits[key] = (min(z_values), max(z_values))
            within_dz = limits[key][1] - limits[key][0]
            maximum_within_dz = max(maximum_within_dz, within_dz)
            if within_dz > self.height_threshold + self.height_epsilon:
                within_columns.add(key)

        cell_radius = int(math.ceil(
            self.neighborhood_radius / self.voxel_size_xy))
        radius_squared = self.neighborhood_radius ** 2 + 1e-12
        for key, (minimum_z, maximum_z) in limits.items():
            for delta_x in range(-cell_radius, cell_radius + 1):
                for delta_y in range(-cell_radius, cell_radius + 1):
                    if delta_x < 0 or (delta_x == 0 and delta_y <= 0):
                        continue
                    distance_squared = self.voxel_size_xy ** 2 * (
                        delta_x * delta_x + delta_y * delta_y)
                    if distance_squared > radius_squared:
                        continue
                    other_key = (key[0] + delta_x, key[1] + delta_y)
                    if other_key not in limits:
                        continue
                    other_minimum, other_maximum = limits[other_key]
                    first_separation = abs(maximum_z - other_minimum)
                    second_separation = abs(other_maximum - minimum_z)
                    maximum_separation = max(first_separation, second_separation)
                    if maximum_separation > (
                            self.height_threshold + self.height_epsilon):
                        if maximum_separation > maximum_neighbor_dz:
                            first_x = (key[0] + 0.5) * self.voxel_size_xy
                            first_y = (key[1] + 0.5) * self.voxel_size_xy
                            other_x = ((other_key[0] + 0.5) *
                                       self.voxel_size_xy)
                            other_y = ((other_key[1] + 0.5) *
                                       self.voxel_size_xy)
                            if first_separation >= second_separation:
                                maximum_neighbor_pair = [
                                    first_x, first_y, maximum_z,
                                    other_x, other_y, other_minimum]
                            else:
                                maximum_neighbor_pair = [
                                    first_x, first_y, minimum_z,
                                    other_x, other_y, other_maximum]
                            maximum_neighbor_dz = maximum_separation
                        triggering_neighbor_pairs += 1
                        neighbor_columns.add(key)
                        neighbor_columns.add(other_key)

        within_indices = {
            index for key in within_columns for index, _ in columns[key]}
        neighbor_indices = {
            index for key in neighbor_columns for index, _ in columns[key]}
        return (within_indices, neighbor_indices, maximum_within_dz,
                maximum_neighbor_dz, triggering_neighbor_pairs,
                maximum_neighbor_pair)

    def robot_pose_at(self, voxel_cloud, admission_cloud):
        values = [float("nan")] * 6
        if (not voxel_cloud.header.frame_id or
                not admission_cloud.header.frame_id):
            return values
        try:
            transform = self.tf_buffer.lookup_transform(
                voxel_cloud.header.frame_id, admission_cloud.header.frame_id,
                voxel_cloud.header.stamp, rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return values

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        sin_roll = 2.0 * (quaternion.w * quaternion.x +
                          quaternion.y * quaternion.z)
        cos_roll = 1.0 - 2.0 * (quaternion.x * quaternion.x +
                                quaternion.y * quaternion.y)
        roll = math.atan2(sin_roll, cos_roll)
        sin_pitch = 2.0 * (quaternion.w * quaternion.y -
                           quaternion.z * quaternion.x)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        sin_yaw = 2.0 * (quaternion.w * quaternion.z +
                         quaternion.x * quaternion.y)
        cos_yaw = 1.0 - 2.0 * (quaternion.y * quaternion.y +
                               quaternion.z * quaternion.z)
        yaw = math.atan2(sin_yaw, cos_yaw)
        return [translation.x, translation.y, translation.z,
                roll, pitch, yaw]

    @staticmethod
    def debug_cloud(header, points, color):
        output_header = Header()
        output_header.stamp = header.stamp
        output_header.frame_id = header.frame_id
        rgb = packed_rgb(*color)
        return point_cloud2.create_cloud(
            output_header, DEBUG_FIELDS,
            [(point[0], point[1], point[2], rgb) for point in points])

    @staticmethod
    def top_bins(points, axis, resolution=0.10, limit=10):
        counts = Counter(
            round(point[axis] / resolution) * resolution for point in points)
        return ";".join(
            "%.2f:%d" % (value, count)
            for value, count in counts.most_common(limit))

    def process(self, voxel_cloud, admission_cloud, ssmi_cloud):
        voxels = self.read_voxels(voxel_cloud)
        admission = list(point_cloud2.read_points(
            admission_cloud,
            field_names=("x", "y", "z", "traversability", "semantic_lable"),
            skip_nans=False))
        ssmi = list(point_cloud2.read_points(
            ssmi_cloud, field_names=("rgb", "semantic_color"), skip_nans=False))

        (within_indices, neighbor_indices, maximum_within_dz,
         maximum_neighbor_dz,
         triggering_neighbor_pairs,
         maximum_neighbor_pair) = self.classify_height_reasons(voxels)
        robot_pose = self.robot_pose_at(voxel_cloud, admission_cloud)
        terrain_indices = {
            index for index, point in enumerate(voxels)
            if int(point[3]) in TERRAIN_LABELS and not math.isfinite(point[4])}
        height_indices = {
            index for index in terrain_indices
            if voxels[index][5] >= self.obstacle_cost - 1e-6}
        height_points = [voxels[index] for index in sorted(height_indices)]
        within_points = [voxels[index] for index in sorted(
            height_indices & within_indices)]
        neighbor_points = [voxels[index] for index in sorted(
            height_indices & neighbor_indices)]

        downstream_indices = [
            index for index, point in enumerate(admission)
            if int(point[4]) in TERRAIN_LABELS and
            point[3] >= self.ssmi_threshold]
        downstream_points = [admission[index] for index in downstream_indices]
        pair_count = min(len(admission), len(ssmi))
        mismatch_points = []
        wall_matches = 0
        wall_mismatches = 0
        for index in downstream_indices:
            if index >= pair_count:
                wall_mismatches += 1
                mismatch_points.append(admission[index])
                continue
            rgb_bits = float_bits(ssmi[index][0])
            semantic_bits = float_bits(ssmi[index][1])
            if rgb_bits == WALL_RGB_BITS and semantic_bits == WALL_RGB_BITS:
                wall_matches += 1
            else:
                wall_mismatches += 1
                mismatch_points.append(admission[index])

        rgb_semantic_disagreements = 0
        expected_color_mismatches = 0
        stair_points = []
        stair_high_cost_points = []
        stair_green_points = []
        stair_wall_points = []
        stair_other_points = []
        stair_wall_without_high_cost = []
        stair_high_cost_not_wall = []
        for index in range(pair_count):
            point = admission[index]
            label = int(point[4])
            high_cost = (math.isfinite(point[3]) and
                         point[3] >= self.ssmi_threshold)
            expected_bits = WALL_RGB_BITS if high_cost else (
                CANONICAL_RGB_BITS[label]
                if 0 <= label < len(CANONICAL_RGB_BITS)
                else UNKNOWN_RGB_BITS)
            rgb_bits = float_bits(ssmi[index][0])
            semantic_bits = float_bits(ssmi[index][1])
            if rgb_bits != semantic_bits:
                rgb_semantic_disagreements += 1
            if rgb_bits != expected_bits or semantic_bits != expected_bits:
                expected_color_mismatches += 1

            if label != STAIR_LABEL:
                continue
            stair_points.append(point)
            if high_cost:
                stair_high_cost_points.append(point)
            if rgb_bits == STAIR_RGB_BITS and semantic_bits == STAIR_RGB_BITS:
                stair_green_points.append(point)
            elif rgb_bits == WALL_RGB_BITS and semantic_bits == WALL_RGB_BITS:
                stair_wall_points.append(point)
                if not high_cost:
                    stair_wall_without_high_cost.append(point)
            else:
                stair_other_points.append(point)
            if high_cost and not (
                    rgb_bits == WALL_RGB_BITS and
                    semantic_bits == WALL_RGB_BITS):
                stair_high_cost_not_wall.append(point)

        # Admission and SSMI snapshots are generated from the same ordered
        # vector. A length mismatch means the unpaired tail cannot be audited.
        expected_color_mismatches += abs(len(admission) - len(ssmi))

        self.height_pub.publish(self.debug_cloud(
            voxel_cloud.header, height_points, (255, 0, 0)))
        self.self_reason_pub.publish(self.debug_cloud(
            voxel_cloud.header, within_points, (255, 128, 0)))
        self.neighbor_reason_pub.publish(self.debug_cloud(
            voxel_cloud.header, neighbor_points, (255, 255, 0)))
        self.downstream_pub.publish(self.debug_cloud(
            admission_cloud.header, downstream_points, (255, 0, 0)))
        self.mismatch_pub.publish(self.debug_cloud(
            admission_cloud.header, mismatch_points, (255, 0, 255)))
        self.stair_green_pub.publish(self.debug_cloud(
            admission_cloud.header, stair_green_points, (152, 251, 152)))
        self.stair_wall_pub.publish(self.debug_cloud(
            admission_cloud.header, stair_wall_points, (255, 0, 0)))
        self.stair_other_pub.publish(self.debug_cloud(
            admission_cloud.header, stair_other_points, (255, 0, 255)))

        coordinates = height_points or [(float("nan"),) * 3]
        stair_wall_coordinates = stair_wall_points or [(float("nan"),) * 3]
        length_mismatch = len(admission) != len(ssmi)
        row = {
            "stamp": "%.9f" % voxel_cloud.header.stamp.to_sec(),
            "voxel_frame": voxel_cloud.header.frame_id,
            "admission_frame": admission_cloud.header.frame_id,
            "voxel_points": len(voxels),
            "terrain_points": len(terrain_indices),
            "height_obstacle_points": len(height_indices),
            "within_column_points": len(height_indices & within_indices),
            "neighbor_column_points": len(height_indices & neighbor_indices),
            "admission_points": len(admission),
            "downstream_height_points": len(downstream_indices),
            "ssmi_points": len(ssmi),
            "ssmi_wall_matches": wall_matches,
            "ssmi_wall_mismatches": wall_mismatches,
            "length_mismatch": int(length_mismatch),
            "height_min_x": min(point[0] for point in coordinates),
            "height_max_x": max(point[0] for point in coordinates),
            "height_min_y": min(point[1] for point in coordinates),
            "height_max_y": max(point[1] for point in coordinates),
            "height_min_z": min(point[2] for point in coordinates),
            "height_max_z": max(point[2] for point in coordinates),
            "maximum_within_column_dz": maximum_within_dz,
            "maximum_neighbor_column_dz": maximum_neighbor_dz,
            "triggering_neighbor_pairs": triggering_neighbor_pairs,
            "maximum_pair_ax": maximum_neighbor_pair[0],
            "maximum_pair_ay": maximum_neighbor_pair[1],
            "maximum_pair_az": maximum_neighbor_pair[2],
            "maximum_pair_bx": maximum_neighbor_pair[3],
            "maximum_pair_by": maximum_neighbor_pair[4],
            "maximum_pair_bz": maximum_neighbor_pair[5],
            "robot_map_x": robot_pose[0],
            "robot_map_y": robot_pose[1],
            "robot_map_z": robot_pose[2],
            "robot_roll": robot_pose[3],
            "robot_pitch": robot_pose[4],
            "robot_yaw": robot_pose[5],
            "top_base_x_bins": self.top_bins(downstream_points, 0),
            "top_base_z_bins": self.top_bins(downstream_points, 2),
            "ssmi_rgb_semantic_disagreements": rgb_semantic_disagreements,
            "ssmi_expected_color_mismatches": expected_color_mismatches,
            "stair_points": len(stair_points),
            "stair_high_cost_points": len(stair_high_cost_points),
            "stair_green_points": len(stair_green_points),
            "stair_wall_points": len(stair_wall_points),
            "stair_other_color_points": len(stair_other_points),
            "stair_wall_without_high_cost": len(stair_wall_without_high_cost),
            "stair_high_cost_not_wall": len(stair_high_cost_not_wall),
            "stair_wall_min_x": min(point[0] for point in stair_wall_coordinates),
            "stair_wall_max_x": max(point[0] for point in stair_wall_coordinates),
            "stair_wall_min_y": min(point[1] for point in stair_wall_coordinates),
            "stair_wall_max_y": max(point[1] for point in stair_wall_coordinates),
            "stair_wall_min_z": min(point[2] for point in stair_wall_coordinates),
            "stair_wall_max_z": max(point[2] for point in stair_wall_coordinates),
            "top_stair_wall_x_bins": self.top_bins(stair_wall_points, 0),
            "top_stair_wall_y_bins": self.top_bins(stair_wall_points, 1),
            "top_stair_wall_z_bins": self.top_bins(stair_wall_points, 2),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.frame_count += 1
        if self.frame_count % self.log_every_n == 0 or wall_mismatches:
            rospy.loginfo(
                "height audit stamp=%.6f: map=%d terrain=%d height=%d "
                "(within=%d neighbor=%d), downstream=%d, SSMI wall=%d/%d, "
                "length_mismatch=%s, max dz=%.2f m, pose=(%.2f, %.2f, %.2f), "
                "base-x bins=[%s], base-z bins=[%s]",
                voxel_cloud.header.stamp.to_sec(), len(voxels),
                len(terrain_indices), len(height_indices),
                len(height_indices & within_indices),
                len(height_indices & neighbor_indices), len(downstream_indices),
                wall_matches, wall_matches + wall_mismatches,
                length_mismatch, maximum_neighbor_dz, robot_pose[0],
                robot_pose[1], robot_pose[2], row["top_base_x_bins"],
                row["top_base_z_bins"])
            rospy.loginfo(
                "SSMI color audit: stair=%d green=%d wall=%d other=%d, "
                "stair_high_cost=%d, wall_without_high_cost=%d, "
                "high_cost_not_wall=%d, rgb/semantic_disagree=%d, "
                "expected_color_mismatch=%d, stair-wall xyz bins=[%s] "
                "[%s] [%s]",
                len(stair_points), len(stair_green_points),
                len(stair_wall_points), len(stair_other_points),
                len(stair_high_cost_points),
                len(stair_wall_without_high_cost),
                len(stair_high_cost_not_wall),
                rgb_semantic_disagreements, expected_color_mismatches,
                row["top_stair_wall_x_bins"],
                row["top_stair_wall_y_bins"],
                row["top_stair_wall_z_bins"])


if __name__ == "__main__":
    rospy.init_node("height_admission_diagnostics")
    HeightAdmissionDiagnostics()
    rospy.spin()
