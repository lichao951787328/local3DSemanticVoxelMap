#!/usr/bin/env python3

"""Record FAR paths and compare them with the active goal in the path frame."""

import csv
import json
import math
import os
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


def rotate_vector(q, vector):
    """Rotate a 3-D vector by a geometry_msgs Quaternion."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    vx, vy, vz = vector
    # v' = v + 2 * qw * (q_xyz x v) + 2 * q_xyz x (q_xyz x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


class PathGoalAlignmentRecorder:
    FIELDNAMES = [
        "record_index", "path_stamp", "goal_stamp", "goal_age_s",
        "path_frame", "goal_frame", "path_pose_count",
        "start_x", "start_y", "start_z",
        "goal_x", "goal_y", "goal_z",
        "end_x", "end_y", "end_z",
        "endpoint_error_xy", "min_path_error_xy", "path_length_xy",
        "goal_range_xy", "endpoint_range_xy", "reaches_goal", "is_stop_path",
    ]

    def __init__(self):
        self.goal_topic = rospy.get_param("~goal_topic", "/way_point")
        self.path_topic = rospy.get_param("~path_topic", "/path")
        self.output_csv = os.path.abspath(
            os.path.expanduser(rospy.get_param(
                "~output_csv", "/tmp/path_goal_alignment.csv")))
        output_root, _ = os.path.splitext(self.output_csv)
        self.output_jsonl = rospy.get_param(
            "~output_jsonl", output_root + "_paths.jsonl")
        self.reach_tolerance = float(rospy.get_param("~reach_tolerance", 0.30))
        self.tf_timeout = rospy.Duration(float(rospy.get_param("~tf_timeout", 0.20)))

        output_directory = os.path.dirname(self.output_csv)
        if output_directory and not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        jsonl_directory = os.path.dirname(os.path.abspath(self.output_jsonl))
        if jsonl_directory and not os.path.isdir(jsonl_directory):
            os.makedirs(jsonl_directory)

        self.csv_file = open(self.output_csv, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.FIELDNAMES)
        self.csv_writer.writeheader()
        self.jsonl_file = open(self.output_jsonl, "w")

        self.lock = threading.Lock()
        self.latest_goal = None
        self.goal_sequence = 0
        self.record_count = 0
        self.stop_count = 0
        self.reach_count = 0
        self.endpoint_error_sum = 0.0
        self.endpoint_error_max = 0.0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher(
            "~alignment_markers", Marker, queue_size=10)
        self.goal_sub = rospy.Subscriber(
            self.goal_topic, PointStamped, self.goal_callback, queue_size=10)
        self.path_sub = rospy.Subscriber(
            self.path_topic, Path, self.path_callback, queue_size=10)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo(
            "Path/goal recorder: path=%s goal=%s CSV=%s full_paths=%s tolerance=%.2f m",
            self.path_topic, self.goal_topic, self.output_csv,
            self.output_jsonl, self.reach_tolerance)

    def goal_callback(self, message):
        with self.lock:
            self.latest_goal = message
            self.goal_sequence += 1

    def transform_goal(self, goal, target_frame, stamp):
        if goal.header.frame_id == target_frame:
            return (goal.point.x, goal.point.y, goal.point.z)

        transform = self.tf_buffer.lookup_transform(
            target_frame, goal.header.frame_id, stamp, self.tf_timeout)
        rotated = rotate_vector(
            transform.transform.rotation,
            (goal.point.x, goal.point.y, goal.point.z))
        translation = transform.transform.translation
        return (
            rotated[0] + translation.x,
            rotated[1] + translation.y,
            rotated[2] + translation.z,
        )

    @staticmethod
    def distance_xy(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def path_callback(self, path):
        with self.lock:
            goal = self.latest_goal
            goal_sequence = self.goal_sequence
        if goal is None:
            rospy.logwarn_throttle(2.0, "Path received before the first goal; skipping")
            return
        if not path.poses:
            rospy.logwarn_throttle(2.0, "Empty path received; skipping")
            return
        if not path.header.frame_id or not goal.header.frame_id:
            rospy.logwarn_throttle(2.0, "Path or goal has an empty frame_id; skipping")
            return

        path_stamp = path.header.stamp
        if path_stamp == rospy.Time(0):
            rospy.logwarn_throttle(2.0, "Path stamp is zero; using latest TF")

        try:
            goal_in_path = self.transform_goal(goal, path.header.frame_id, path_stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn_throttle(
                2.0, "Cannot transform goal %s -> %s at %.9f: %s",
                goal.header.frame_id, path.header.frame_id,
                path_stamp.to_sec(), error)
            return

        points = [
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            for pose in path.poses
        ]
        start = points[0]
        endpoint = points[-1]
        endpoint_error = self.distance_xy(endpoint, goal_in_path)
        min_path_error = min(
            self.distance_xy(point, goal_in_path) for point in points)
        path_length = sum(
            self.distance_xy(points[index - 1], points[index])
            for index in range(1, len(points)))
        goal_range = math.hypot(goal_in_path[0], goal_in_path[1])
        endpoint_range = math.hypot(endpoint[0], endpoint[1])
        is_stop_path = len(points) <= 1 or path_length < 1e-3
        reaches_goal = min_path_error <= self.reach_tolerance
        goal_age = (path_stamp - goal.header.stamp).to_sec()

        with self.lock:
            self.record_count += 1
            record_index = self.record_count
            if is_stop_path:
                self.stop_count += 1
            else:
                self.endpoint_error_sum += endpoint_error
                self.endpoint_error_max = max(self.endpoint_error_max, endpoint_error)
                if reaches_goal:
                    self.reach_count += 1

            row = {
                "record_index": record_index,
                "path_stamp": "%.9f" % path_stamp.to_sec(),
                "goal_stamp": "%.9f" % goal.header.stamp.to_sec(),
                "goal_age_s": "%.6f" % goal_age,
                "path_frame": path.header.frame_id,
                "goal_frame": goal.header.frame_id,
                "path_pose_count": len(points),
                "start_x": "%.6f" % start[0],
                "start_y": "%.6f" % start[1],
                "start_z": "%.6f" % start[2],
                "goal_x": "%.6f" % goal_in_path[0],
                "goal_y": "%.6f" % goal_in_path[1],
                "goal_z": "%.6f" % goal_in_path[2],
                "end_x": "%.6f" % endpoint[0],
                "end_y": "%.6f" % endpoint[1],
                "end_z": "%.6f" % endpoint[2],
                "endpoint_error_xy": "%.6f" % endpoint_error,
                "min_path_error_xy": "%.6f" % min_path_error,
                "path_length_xy": "%.6f" % path_length,
                "goal_range_xy": "%.6f" % goal_range,
                "endpoint_range_xy": "%.6f" % endpoint_range,
                "reaches_goal": int(reaches_goal),
                "is_stop_path": int(is_stop_path),
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()

            full_record = dict(row)
            full_record["goal_sequence"] = goal_sequence
            full_record["path_points"] = [list(point) for point in points]
            self.jsonl_file.write(json.dumps(full_record, separators=(",", ":")) + "\n")
            self.jsonl_file.flush()

        self.publish_markers(path.header.frame_id, path_stamp, goal_in_path, endpoint)

        if record_index % 50 == 0:
            moving_count = record_index - self.stop_count
            mean_error = (self.endpoint_error_sum / moving_count
                          if moving_count else float("nan"))
            rospy.loginfo(
                "Recorded %d paths: moving=%d, within %.2f m=%d, "
                "mean endpoint error=%.3f m, max=%.3f m",
                record_index, moving_count, self.reach_tolerance,
                self.reach_count, mean_error, self.endpoint_error_max)

    def make_marker(self, frame, stamp, marker_id, marker_type, color):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = "path_goal_alignment"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = rospy.Duration(0.5)
        return marker

    def publish_markers(self, frame, stamp, goal, endpoint):
        goal_marker = self.make_marker(
            frame, stamp, 0, Marker.SPHERE, (0.1, 1.0, 0.1, 0.9))
        goal_marker.pose.position.x = goal[0]
        goal_marker.pose.position.y = goal[1]
        goal_marker.pose.position.z = goal[2]
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.35
        self.marker_pub.publish(goal_marker)

        endpoint_marker = self.make_marker(
            frame, stamp, 1, Marker.SPHERE, (1.0, 0.2, 0.1, 0.9))
        endpoint_marker.pose.position.x = endpoint[0]
        endpoint_marker.pose.position.y = endpoint[1]
        endpoint_marker.pose.position.z = endpoint[2]
        endpoint_marker.scale.x = endpoint_marker.scale.y = endpoint_marker.scale.z = 0.30
        self.marker_pub.publish(endpoint_marker)

        error_marker = self.make_marker(
            frame, stamp, 2, Marker.LINE_LIST, (1.0, 0.9, 0.1, 0.9))
        error_marker.scale.x = 0.06
        error_marker.points = [
            Point(x=endpoint[0], y=endpoint[1], z=endpoint[2]),
            Point(x=goal[0], y=goal[1], z=goal[2]),
        ]
        self.marker_pub.publish(error_marker)

    def shutdown(self):
        with self.lock:
            if self.csv_file.closed:
                return
            moving_count = self.record_count - self.stop_count
            mean_error = (self.endpoint_error_sum / moving_count
                          if moving_count else float("nan"))
            rospy.loginfo(
                "Path/goal result: total=%d moving=%d stop=%d within %.2f m=%d "
                "mean endpoint error=%.3f m max=%.3f m. Files: %s, %s",
                self.record_count, moving_count, self.stop_count,
                self.reach_tolerance, self.reach_count, mean_error,
                self.endpoint_error_max, self.output_csv, self.output_jsonl)
            self.csv_file.close()
            self.jsonl_file.close()


def main():
    rospy.init_node("path_goal_alignment_recorder")
    PathGoalAlignmentRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
