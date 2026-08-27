#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class DeepReachPickNode(Node):
    def __init__(self):
        super().__init__('deep_reach_pick_node')
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

    def execute_pick(self):
        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Waypoint 1: Raise & Prepare (Hover High)
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, -0.7, 1.6, -2.47, -1.5708, 0.0]
        p1.time_from_start.sec = 3

        # Waypoint 2: Mid-Descent over Block
        p2 = JointTrajectoryPoint()
        p2.positions = [0.0, -0.45, 1.85, -2.97, -1.5708, 0.0]
        p2.time_from_start.sec = 6

        # Waypoint 3: Full Deep Reach onto Top of Block
        p3 = JointTrajectoryPoint()
        p3.positions = [0.0, -0.30, 2.0, -3.12, -1.5708, 0.0]
        p3.time_from_start.sec = 9

        msg.points = [p1, p2, p3]
        self.get_logger().info("--> Lowering arm completely onto the block...")
        self.arm_pub.publish(msg)

        # Wait for full reach contact
        time.sleep(9.5)

        # Trigger Grasp Attachment
        self.get_logger().info("--> Direct contact established! Triggering grasp...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        # Waypoint 4: Smooth Lift Upwards with Object
        lift_msg = JointTrajectory()
        lift_msg.joint_names = msg.joint_names
        
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, -1.0, 1.2, -1.77, -1.5708, 0.0]
        p4.time_from_start.sec = 4

        lift_msg.points = [p4]
        self.get_logger().info("--> Carrying object upward...")
        self.arm_pub.publish(lift_msg)

def main():
    rclpy.init()
    node = DeepReachPickNode()
    node.execute_pick()
    time.sleep(5)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
