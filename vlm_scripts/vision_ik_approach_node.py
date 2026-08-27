#!/usr/bin/env python3
"""
Vision-based IK Approach Node
------------------------------
Replaces the hardcoded joint-angle guessing in arm_approach_node.py with a
real perception-driven pipeline:

  1. Read the object's pixel bounding box from /gripper_camera/fragility_analysis
     (the VLM now reports bbox_center_x/y and bbox_width_px).
  2. Estimate the object's 3D position relative to the camera using the
     pinhole camera model: known apple diameter + camera focal length
     (derived from the camera's horizontal_fov and image width in the URDF)
     gives an approximate distance; pixel offset from image center gives
     lateral/vertical offset at that distance.
  3. Use tf2 to transform that 3D point from the camera frame into base_link.
  4. Use ikpy (loaded from the same expanded URDF) to solve for the 6 arm
     joint angles that reach that point.
  5. Publish the resulting trajectory to joint_trajectory_controller.

CALIBRATION NOTE: step 2's sign conventions (which pixel axis maps to which
camera-frame axis) depend on how the camera link is mounted and how Gazebo's
camera plugin orients its image. The defaults below assume a standard
forward-looking camera (local +X forward, +Y left, +Z up, matching the
gripper_camera_link's URDF axes -- NOT a ROS optical frame convention).
If the arm consistently moves in the wrong direction relative to the apple,
flip the sign on X_SIGN / Y_SIGN / Z_SIGN below and re-test.
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from ikpy.chain import Chain

FRAGILITY_TOPIC = "/gripper_camera/fragility_analysis"
ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"
URDF_PATH = "/home/tt501/dexproject/vlm_scripts/expanded_robot.urdf"

CAMERA_FRAME = "gripper_camera_link"
BASE_FRAME = "base_link"

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# --- Camera intrinsics, derived from the URDF sensor definition ---
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
HORIZONTAL_FOV_RAD = 1.047  # from <horizontal_fov> in ur5e_dexhand.xacro
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)

# Known real-world object size used for distance estimation (meters).
KNOWN_APPLE_DIAMETER_M = 0.08

# Axis sign conventions -- flip these during calibration if the arm moves
# the wrong direction. See CALIBRATION NOTE above.
X_SIGN = 1.0   # forward (distance from camera)
Y_SIGN = -1.0  # left/right (pixel x increases right, but +Y is often left)
Z_SIGN = -1.0  # up/down (pixel y increases downward, but +Z is up)

# Stand off this far (meters) short of the estimated apple center along the
# approach direction, so the fingers arrive just short of the apple rather
# than trying to place the wrist exactly inside it.
APPROACH_STANDOFF_M = 0.10

MOVE_DURATION_SEC = 4.0


class VisionIKApproachNode(Node):
    def __init__(self):
        super().__init__("vision_ik_approach_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base_link"])
        self.get_logger().info(f"ikpy chain loaded with {len(self.chain.links)} links.")

        # Mask: only the 6 revolute arm joints (indices 2-7) are active;
        # the fixed base links are not solved for.
        self.active_mask = [False] * len(self.chain.links)
        for i in range(2, 8):
            if i < len(self.active_mask):
                self.active_mask[i] = True
        self.chain.active_links_mask = self.active_mask

        self.traj_pub = self.create_publisher(JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        self.create_subscription(String, FRAGILITY_TOPIC, self.on_fragility_msg, 10)

        self._last_move_time = 0.0
        self._min_seconds_between_moves = 5.0  # avoid spamming new goals every VLM message

        self.get_logger().info("Vision IK approach node ready.")

    def on_fragility_msg(self, msg: String):
        try:
            analysis = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        obj = analysis.get("object_name", "unknown object")
        if obj == "none visible":
            return

        bbox_w = analysis.get("bbox_width_px", 0)
        if not bbox_w or bbox_w <= 0:
            self.get_logger().warning("No usable bbox_width_px in analysis, skipping.")
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_move_time < self._min_seconds_between_moves:
            return  # already approaching / just moved, don't spam new goals
        self._last_move_time = now

        px = analysis.get("bbox_center_x", IMAGE_WIDTH_PX / 2)
        py = analysis.get("bbox_center_y", IMAGE_HEIGHT_PX / 2)

        target_point_camera = self.estimate_3d_position(px, py, bbox_w)
        if target_point_camera is None:
            return

        target_point_base = self.transform_to_base(target_point_camera)
        if target_point_base is None:
            return

        self.get_logger().info(
            f"Estimated '{obj}' at base_link position "
            f"({target_point_base[0]:.3f}, {target_point_base[1]:.3f}, {target_point_base[2]:.3f})"
        )

        joint_angles = self.solve_ik(target_point_base)
        if joint_angles is None:
            return

        self.publish_trajectory(joint_angles)

    def estimate_3d_position(self, px, py, bbox_w_px):
        """Pinhole model: known object size + apparent pixel width -> distance,
        then pixel offset from image center -> lateral/vertical offset."""
        if bbox_w_px <= 0:
            return None

        distance_m = (KNOWN_APPLE_DIAMETER_M * FOCAL_LENGTH_PX) / bbox_w_px

        dx_px = px - (IMAGE_WIDTH_PX / 2.0)
        dy_px = py - (IMAGE_HEIGHT_PX / 2.0)

        lateral_m = (dx_px * distance_m) / FOCAL_LENGTH_PX
        vertical_m = (dy_px * distance_m) / FOCAL_LENGTH_PX

        x = X_SIGN * distance_m
        y = Y_SIGN * lateral_m
        z = Z_SIGN * vertical_m

        self.get_logger().info(
            f"Pinhole estimate: distance={distance_m:.3f}m, "
            f"camera-frame point=({x:.3f}, {y:.3f}, {z:.3f})"
        )
        return (x, y, z)

    def transform_to_base(self, point_camera):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = CAMERA_FRAME
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x = point_camera[0]
            point_stamped.point.y = point_camera[1]
            point_stamped.point.z = point_camera[2]

            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None

    def solve_ik(self, target_base):
        # Pull back along the approach direction so we stop short of the
        # apple center rather than driving the wrist into it.
        direction = np.array(target_base)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])

        standoff_target = np.array(target_base) - direction * APPROACH_STANDOFF_M

        try:
            ik_solution = self.chain.inverse_kinematics(standoff_target)
        except Exception as e:
            self.get_logger().error(f"IK solve failed: {e}")
            return None

        # ik_solution includes all chain links (fixed + active); extract just
        # the 6 active arm joint angles at indices 2-7.
        arm_angles = [ik_solution[i] for i in range(2, 8)]
        self.get_logger().info(f"IK solution (arm joints): {arm_angles}")
        return arm_angles

    def publish_trajectory(self, joint_angles):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(MOVE_DURATION_SEC)
        point.time_from_start.nanosec = int((MOVE_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.traj_pub.publish(traj)
        self.get_logger().info("Published IK-solved approach trajectory.")


def main(args=None):
    rclpy.init(args=args)
    node = VisionIKApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
