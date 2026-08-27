#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class FixedPosePickerNode(Node):
    def __init__(self):
        super().__init__('fixed_pose_picker_node')
        
        # Controller publisher for UR5e arm joints
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Attach publisher for Gazebo grasp lock
        self.attach_pub = self.create_publisher(
            Empty,
            '/attach',
            10
        )
        
        # UR5e Base & Link Kinematics (Meters)
        self.a2 = 0.425          # Upper arm length
        self.a3 = 0.3922         # Forearm length
        self.z_shoulder = 0.6075 # Ground-to-shoulder height (base mount included)
        self.hand_length = 0.18  # Tool0 to palm/fingertip reach offset
        
        time.sleep(1.5)

    def compute_palm_down_ik(self, target_x, target_y, target_z_top):
        """
        Calculates exact UR5e joint angles while forcing the 
        palm to point STRAIGHT DOWN (-Z axis) towards the ground.
        """
        r = np.sqrt(target_x**2 + target_y**2)
        pan = np.arctan2(target_y, target_x)
        
        # Target wrist Z position to keep hand above object
        z_wrist = target_z_top + self.hand_length
        
        dx = r
        dz = z_wrist - self.z_shoulder
        D = np.sqrt(dx**2 + dz**2)
        
        # Elbow angle (elbow-down configuration)
        cos_elbow = (D**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow = np.arccos(cos_elbow)
        
        # Shoulder lift angle
        gamma = np.arctan2(dz, dx)
        cos_alpha = (self.a2**2 + D**2 - self.a3**2) / (2 * self.a2 * D)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        phi_upper = gamma - alpha
        shoulder_lift = phi_upper - (np.pi / 2.0)
        
        # Forearm angle relative to ground horizontal
        phi_forearm = phi_upper + elbow
        
        # Keep tool axis pointing straight DOWN (-90 deg / -pi/2)
        wrist_1 = -np.pi / 2.0 - phi_forearm
        
        # Zero-out wrist roll and yaw so hand stays square with ground
        wrist_2 = 0.0
        wrist_3 = 0.0
        
        return [
            float(pan),
            float(shoulder_lift),
            float(elbow),
            float(wrist_1),
            float(wrist_2),
            float(wrist_3)
        ]

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

    def execute_pick(self, block_x, block_y, block_height):
        # Top surface of the block sitting on ground
        z_block_top = block_height  # 0.14m
        
        self.get_logger().info("--> Calculating Straight Palm-Down Trajectories...")
        
        # 1. Staging pose (Home position)
        home_pose = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
        
        # 2. Overhead Hover Pose (15cm directly above block top)
        approach_pose = self.compute_palm_down_ik(block_x, block_y, z_block_top + 0.15)
        
        # 3. Direct Contact Pose (Hand lowers straight down onto block top)
        grasp_pose = self.compute_palm_down_ik(block_x, block_y, z_block_top)

        # Step 1: Return to neutral
        self.get_logger().info("1. Moving to neutral home position...")
        self.send_arm_trajectory(home_pose, duration_sec=3.0)
        time.sleep(3.5)

        # Step 2: Hover overhead
        self.get_logger().info("2. Positioning hand directly overhead (Palm pointing DOWN)...")
        self.send_arm_trajectory(approach_pose, duration_sec=3.5)
        time.sleep(4.0)

        # Step 3: Descend straight down
        self.get_logger().info("3. Lowering down directly onto block top surface...")
        self.send_arm_trajectory(grasp_pose, duration_sec=2.5)
        time.sleep(3.0)

        # Step 4: Lock/Grasp object
        self.get_logger().info("4. Engaging grasp attach lock...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        # Step 5: Lift straight up
        self.get_logger().info("5. Retracting arm straight UP with block...")
        self.send_arm_trajectory(approach_pose, duration_sec=2.5)
        time.sleep(3.0)
        
        self.send_arm_trajectory(home_pose, duration_sec=3.5)
        time.sleep(4.0)

        self.get_logger().info("SUCCESS: Block fully picked and lifted!")

def main():
    rclpy.init()
    node = FixedPosePickerNode()
    
    # Target: Block centered at X=0.65m, Y=0.0m, with height = 0.14m
    node.execute_pick(block_x=0.65, block_y=0.0, block_height=0.14)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
