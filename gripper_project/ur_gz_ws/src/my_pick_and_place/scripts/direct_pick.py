#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class DirectPickNode(Node):
    def __init__(self):
        super().__init__('direct_pick_node')
        
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
        time.sleep(1.0)

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

    def run_sequence(self):
        # 1. Neutral Staging Pose (Hand pointing forward/down)
        home_pose    = [0.0, -1.5708, 0.0, -1.5708, -1.5708, 0.0]
        
        # 2. Hover Overhead (X=0.50m, Z=0.28m, Palm pointing STRAIGHT DOWN)
        hover_pose   = [0.0, -0.8066, 1.8216, -0.5133, -1.5708, 0.0]
        
        # 3. Contact Top Surface (X=0.50m, Z=0.14m - Top of 14cm block)
        contact_pose = [0.0, -1.1172, 1.7194, -0.3049, -1.5708, 0.0]

        self.get_logger().info("1. Moving to neutral staging pose...")
        self.send_arm_trajectory(home_pose, duration_sec=3.0)
        time.sleep(3.5)

        self.get_logger().info("2. Reaching straight ahead over block (Palm facing DOWN)...")
        self.send_arm_trajectory(hover_pose, duration_sec=3.5)
        time.sleep(4.0)

        self.get_logger().info("3. Lowering hand directly onto block top surface...")
        self.send_arm_trajectory(contact_pose, duration_sec=2.5)
        time.sleep(3.0)

        self.get_logger().info("4. Engaging grasp attach lock...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        self.get_logger().info("5. Lifting block straight UP...")
        self.send_arm_trajectory(hover_pose, duration_sec=2.5)
        time.sleep(3.0)

        self.send_arm_trajectory(home_pose, duration_sec=3.5)
        time.sleep(4.0)

        self.get_logger().info("SUCCESS: Block picked and lifted!")

def main():
    rclpy.init()
    node = DirectPickNode()
    node.run_sequence()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
