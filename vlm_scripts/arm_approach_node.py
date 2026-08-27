#!/usr/bin/env python3
"""
Simple Arm Approach Node
-------------------------
Moves the UR5e arm through a sequence of joint-space waypoints (home -> hover
over the apple -> descend) so the hand ends up positioned over an apple
before adaptive_grasp_controller.py takes over the finger squeeze.

This project has no MoveIt / IK solver running -- just direct joint control
via joint_trajectory_controller. So instead of computing poses from apple
XYZ coordinates, this script replays a fixed sequence of joint angles you
tune once by jogging the arm manually and reading back /joint_states.

HOW TO GET REAL WAYPOINT VALUES (do this once):
  1. With the sim running, open a terminal and watch positions live:
       ros2 topic echo /joint_states --field position
  2. Publish small test trajectories by hand (see example command at the
     bottom of this file) to jog the arm to a pose that hovers directly
     above the first apple in the row.
  3. Copy the 6 arm joint positions you land on into HOVER_POSE below.
  4. Repeat for a lower "descend" pose that puts the hand close enough to
     the apple for the fingers to actually make contact when they close.
  5. Save, then run this node.

The values below are PLACEHOLDERS (arm's default spawn pose, repeated) --
the arm will not visibly move anywhere useful until you replace them.
"""
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

HOME_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
HOVER_POSE = [0.3, -0.7, 1.1, -1.97, -1.57, 0.0]
DESCEND_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]    # TODO: replace

MOVE_DURATION_SEC = 3.0


class ArmApproachNode(Node):
    def __init__(self):
        super().__init__("arm_approach_node")
        self.pub = self.create_publisher(JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        self.get_logger().info("Arm approach node ready.")
        self.create_timer(1.0, self.run_sequence_once)
        self._done = False

    def send_waypoint(self, positions, duration_sec):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj.points.append(point)
        self.pub.publish(traj)
        self.get_logger().info(f"Sent waypoint: {positions}")

    def run_sequence_once(self):
        if self._done:
            return
        self._done = True

        self.get_logger().info("Moving to HOME_POSE...")
        self.send_waypoint(HOME_POSE, MOVE_DURATION_SEC)
        time.sleep(MOVE_DURATION_SEC + 0.5)

        self.get_logger().info("Moving to HOVER_POSE...")
        self.send_waypoint(HOVER_POSE, MOVE_DURATION_SEC)
        time.sleep(MOVE_DURATION_SEC + 0.5)

        self.get_logger().info("Moving to DESCEND_POSE..."); return  # TEMP: disabled for tuning
        self.send_waypoint(DESCEND_POSE, MOVE_DURATION_SEC)
        time.sleep(MOVE_DURATION_SEC + 0.5)

        self.get_logger().info(
            "Approach sequence complete. Hand should now be positioned over "
            "the apple -- start adaptive_grasp_controller.py now if it isn't "
            "already running."
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArmApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
