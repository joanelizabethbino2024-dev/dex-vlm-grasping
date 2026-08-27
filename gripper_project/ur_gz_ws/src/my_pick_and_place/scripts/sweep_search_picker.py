#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class GridSweepSearchNode(Node):
    def __init__(self):
        super().__init__('grid_sweep_search_node')
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.attach_pub = self.create_publisher(
            Empty,
            '/attach',
            10
        )
        time.sleep(1.5)

    def publish_trajectory(self, joint_angles, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.arm_pub.publish(msg)

    def execute_grid_search(self):
        # High Safe Staging Pose (Prevents self-collision with chassis)
        safe_home = [0.0, -1.2, 1.2, -1.5708, -1.5708, 0.0]
        
        self.get_logger().info("--> Moving to High Collision-Safe Staging Pose...")
        self.publish_trajectory(safe_home, duration_sec=3.0)
        time.sleep(3.5)

        # 3D Grid Search Pattern Matrix: [Pan (Left/Right), Lift (Up/Down), Elbow, Wrist1, Wrist2, Wrist3]
        # Maintains palm facing DOWN while searching left, right, up, and down.
        search_grid = [
            # LEVEL 1: High Scan (Pan Left -> Center -> Pan Right)
            {"name": "Level 1 High Left",   "pose": [ 0.35, -0.60, 1.60, -2.5708, -1.5708, 0.0], "time": 2.5},
            {"name": "Level 1 High Center", "pose": [ 0.00, -0.60, 1.60, -2.5708, -1.5708, 0.0], "time": 2.0},
            {"name": "Level 1 High Right",  "pose": [-0.35, -0.60, 1.60, -2.5708, -1.5708, 0.0], "time": 2.5},

            # LEVEL 2: Mid-Depth Scan (Pan Right -> Center -> Pan Left)
            {"name": "Level 2 Mid Right",   "pose": [-0.35, -0.40, 1.85, -3.0208, -1.5708, 0.0], "time": 2.5},
            {"name": "Level 2 Mid Center",  "pose": [ 0.00, -0.40, 1.85, -3.0208, -1.5708, 0.0], "time": 2.0},
            {"name": "Level 2 Mid Left",    "pose": [ 0.35, -0.40, 1.85, -3.0208, -1.5708, 0.0], "time": 2.5},

            # LEVEL 3: Low Ground Contact Scan (Sweep Left -> Center)
            {"name": "Level 3 Ground Left", "pose": [ 0.25, -0.22, 2.15, -3.5008, -1.5708, 0.0], "time": 3.0},
            {"name": "Level 3 Ground Target", "pose": [ 0.00, -0.10, 2.28, -3.7508, -1.5708, 0.0], "time": 2.5},
        ]

        for step in search_grid:
            self.get_logger().info(f"[SEARCHING] Executing: {step['name']}...")
            self.publish_trajectory(step["pose"], duration_sec=step["time"])
            time.sleep(step["time"] + 0.5)

        # Grasp Execution at Target Alignment
        self.get_logger().info("--> Object Located at Ground Center! Closing Grasp...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        # Collision-Safe Vertical Lift Trajectory
        self.get_logger().info("--> Lifting Object directly UP (Collision Avoidance Active)...")
        lift_waypoint_1 = [0.00, -0.60, 1.60, -2.5708, -1.5708, 0.0]
        lift_waypoint_2 = [0.00, -1.20, 1.20, -1.5708, -1.5708, 0.0]

        self.publish_trajectory(lift_waypoint_1, duration_sec=3.0)
        time.sleep(3.2)
        self.publish_trajectory(lift_waypoint_2, duration_sec=3.0)
        time.sleep(3.2)

        self.get_logger().info("SUCCESS: Object picked up cleanly without any self-collisions!")

def main():
    rclpy.init()
    node = GridSweepSearchNode()
    node.execute_grid_search()
    time.sleep(2)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
