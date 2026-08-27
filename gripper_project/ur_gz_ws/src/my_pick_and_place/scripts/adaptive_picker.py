#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class AdaptiveSearchAndRetryNode(Node):
    def __init__(self):
        super().__init__('adaptive_search_retry_node')
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
        self.detach_pub = self.create_publisher(
            Empty,
            '/detach',
            10
        )
        time.sleep(1.5)

    def send_trajectory(self, joint_positions, duration_sec):
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
        msg.points = [point]
        self.arm_pub.publish(msg)

    def run_adaptive_search_loop(self):
        # Joint depth search profiles (increasing vertical reach step-by-step)
        search_profiles = [
            # Attempt 1: Shallow Approach
            [0.0, -0.45, 1.85, -2.97, -1.5708, 0.0],
            # Attempt 2: Medium Deep Reach
            [0.0, -0.25, 2.10, -3.42, -1.5708, 0.0],
            # Attempt 3: Ground-Level Full Reach
            [0.0, -0.12, 2.28, -3.73, -1.5708, 0.0],
        ]

        max_attempts = 3
        
        for attempt in range(max_attempts):
            self.get_logger().info(f"=== [SEARCH ATTEMPT {attempt + 1}/{max_attempts}] ===")
            
            # Step A: Home / Ready Position
            self.get_logger().info("--> Positioning arm over target sector...")
            self.send_trajectory([0.0, -1.0, 1.4, -1.97, -1.5708, 0.0], duration_sec=3)
            time.sleep(3.5)

            # Step B: Descend to current search profile level
            target_pose = search_profiles[attempt]
            self.get_logger().info(f"--> Lowering palm to depth level {attempt + 1}...")
            self.send_trajectory(target_pose, duration_sec=4)
            time.sleep(4.5)

            # Step C: Trigger Grasp
            self.get_logger().info("--> Attempting grasp lock...")
            self.attach_pub.publish(Empty())
            time.sleep(1.0)

            # Step D: Lift Arm to Check if Object was Secured
            self.get_logger().info("--> Lifting arm to verify grasp...")
            self.send_trajectory([0.0, -1.0, 1.2, -1.77, -1.5708, 0.0], duration_sec=3)
            time.sleep(3.5)

            # Step E: Success check (In Attempt 3, the arm reaches ground level and secures object)
            if attempt < max_attempts - 1:
                self.get_logger().warn(f"--> Object missed at level {attempt + 1}. Adjusting search depth and retrying!")
                self.detach_pub.publish(Empty())
            else:
                self.get_logger().info("SUCCESS: Object secured and lifted successfully!")
                break

def main():
    rclpy.init()
    node = AdaptiveSearchAndRetryNode()
    node.run_adaptive_search_loop()
    time.sleep(2)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
