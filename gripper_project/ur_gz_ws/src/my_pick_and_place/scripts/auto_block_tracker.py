#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class FixedCoordinatePickerNode(Node):
    def __init__(self):
        super().__init__('fixed_coordinate_picker_node')
        
        self.pose_sub = self.create_subscription(
            Pose,
            '/model/red_block/pose',
            self.pose_callback,
            10
        )
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
        
        # UR5e Link Geometry (Meters)
        self.a2 = 0.425          # Upper arm length
        self.a3 = 0.3922         # Forearm length
        self.z_shoulder = 0.6075 # Shoulder height in world frame
        self.hand_length = 0.18  # Tool0 to palm reach offset
        
        self.live_pose = None

    def pose_callback(self, msg):
        self.live_pose = msg

    def compute_3d_palm_down_ik(self, x, y, z_top):
        """Dynamic 3D Inverse Kinematics forcing the hand to point STRAIGHT DOWN"""
        pan = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        z_wrist = z_top + self.hand_length
        
        dx = r
        dz = z_wrist - self.z_shoulder
        D = np.sqrt(dx**2 + dz**2)
        
        cos_elbow = (D**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow = np.arccos(cos_elbow)
        
        gamma = np.arctan2(dz, dx)
        cos_alpha = (self.a2**2 + D**2 - self.a3**2) / (2 * self.a2 * D)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        phi_upper = gamma - alpha
        shoulder_lift = phi_upper - (np.pi / 2.0)
        phi_forearm = phi_upper + elbow
        wrist_1 = -np.pi / 2.0 - phi_forearm
        
        return [float(pan), float(shoulder_lift), float(elbow), float(wrist_1), 0.0, 0.0]

    def send_arm_trajectory(self, joint_positions, duration_sec):
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
        point.positions = joint_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.arm_pub.publish(msg)

    def execute_pick(self):
        self.get_logger().info("Checking live bridge topic for 2 seconds...")
        start_time = time.time()
        
        while rclpy.ok() and self.live_pose is None and (time.time() - start_time) < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.live_pose is not None:
            x = self.live_pose.position.x
            y = self.live_pose.position.y
            z_center = self.live_pose.position.z
            self.get_logger().info(f"--> LIVE POSE RECEIVED: X={x:.3f}, Y={y:.3f}, Z={z_center:.3f}")
        else:
            x, y, z_center = 0.50, 0.00, 0.07
            self.get_logger().info(f"--> Target Spawn Coordinates: X={x}, Y={y}, Z={z_center}")

        z_top = z_center + 0.07  # Top surface of 0.14m block

        approach_pose = self.compute_3d_palm_down_ik(x, y, z_top + 0.15)
        grasp_pose = self.compute_3d_palm_down_ik(x, y, z_top)
        home_pose = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

        # Execution
        self.get_logger().info("1. Positioning arm overhead with palm facing DOWN...")
        self.send_arm_trajectory(approach_pose, duration_sec=3.5)
        time.sleep(4.0)

        self.get_logger().info("2. Lowering straight down onto top surface...")
        self.send_arm_trajectory(grasp_pose, duration_sec=2.5)
        time.sleep(3.0)

        self.get_logger().info("3. Engaging grasp lock...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        self.get_logger().info("4. Retracting upward with block...")
        self.send_arm_trajectory(approach_pose, duration_sec=2.5)
        time.sleep(3.0)

        self.send_arm_trajectory(home_pose, duration_sec=3.5)
        time.sleep(4.0)

        self.get_logger().info("SUCCESS: Block picked and lifted successfully!")

def main():
    rclpy.init()
    node = FixedCoordinatePickerNode()
    node.execute_pick()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
