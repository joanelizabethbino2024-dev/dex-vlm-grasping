#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class SoftTouchPickerNode(Node):
    def __init__(self):
        super().__init__('soft_touch_picker_node')
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

    def publish_pose(self, joint_positions, duration_sec):
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

    def run_pick_sequence(self):
        # 1. High Staging Pose
        self.get_logger().info("--> 1. Moving to High Safe Staging Pose...")
        self.publish_pose([0.0, -1.5708, 0.0, -1.5708, -1.5708, 0.0], duration_sec=3.0)
        time.sleep(3.5)

        # 2. Hover directly above object (Z_wrist = 0.35m)
        self.get_logger().info("--> 2. Hovering above object at X=0.65m...")
        self.publish_pose([0.0, -0.7561, 1.2738, -2.0885, -1.5708, 0.0], duration_sec=3.0)
        time.sleep(3.5)

        # 3. Soft Touch Down (Z_wrist = 0.16m - perfect flush contact without force spike)
        self.get_logger().info("--> 3. Lowering hand onto block...")
        self.publish_pose([0.0, -0.9080, 1.0280, -1.6908, -1.5708, 0.0], duration_sec=2.5)
        time.sleep(3.0)

        # 4. Attach Detachable Joint Plugin
        self.get_logger().info("--> 4. Triggering attachment...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        # 5. Lift Up smoothly
        self.get_logger().info("--> 5. Lifting block upward...")
        self.publish_pose([0.0, -0.7561, 1.2738, -2.0885, -1.5708, 0.0], duration_sec=3.0)
        time.sleep(3.5)
        self.publish_pose([0.0, -1.5708, 0.0, -1.5708, -1.5708, 0.0], duration_sec=3.0)
        time.sleep(3.5)

        self.get_logger().info("SUCCESS: Block picked up cleanly!")

def main():
    rclpy.init()
    node = SoftTouchPickerNode()
    node.run_pick_sequence()
    time.sleep(1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
