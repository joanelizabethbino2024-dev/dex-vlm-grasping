#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

class MoveItPickPlaceNode(Node):
    def __init__(self):
        super().__init__('moveit_pick_place_node')
        
        # Force ROS 2 Node to use Gazebo Sim Time
        if not self.has_parameter('use_sim_time'):
             self.declare_parameter('use_sim_time', True)
        
        # MoveIt IK Service Client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        # Arm Controller & Attach Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.attach_pub = self.create_publisher(Empty, '/attach', 10)

        self.get_logger().info("Waiting for MoveIt '/compute_ik' service...")
        while not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for MoveIt IK service...")

    def solve_ik(self, x, y, z):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 1

        # Seed State
        seed_state = JointState()
        seed_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        seed_state.position = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        req.ik_request.robot_state.joint_state = seed_state

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = float(x)
        target_pose.pose.position.y = float(y)
        target_pose.pose.position.z = float(z)

        # Orientation: Palm pointing straight down (-Z)
        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0

        req.ik_request.pose_stamped = target_pose

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res is not None and res.error_code.val == 1:
            return res.solution.joint_state
        else:
            self.get_logger().error(f"IK Failed for ({x}, {y}, {z})")
            return None

    def send_arm_trajectory(self, joint_state, duration_sec):
        msg = JointTrajectory()
        # Attach current sim-time timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(joint_state.name)
        
        point = JointTrajectoryPoint()
        point.positions = list(joint_state.position)
        point.velocities = [0.0] * len(joint_state.position)
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        
        msg.points = [point]
        self.arm_pub.publish(msg)

    def execute_pick(self):
        block_x, block_y = 0.50, 0.00
        hover_z = 0.35
        contact_z = 0.18

        self.get_logger().info("Computing MoveIt IK solutions...")
        hover_state = self.solve_ik(block_x, block_y, hover_z)
        contact_state = self.solve_ik(block_x, block_y, contact_z)

        if hover_state is None or contact_state is None:
            self.get_logger().error("Aborting: Could not solve IK.")
            return

        # 1. Hover Overhead
        self.get_logger().info("1. Positioning overhead using MoveIt solution...")
        self.send_arm_trajectory(hover_state, duration_sec=3.5)
        time.sleep(4.0)

        # 2. Lower to Object
        self.get_logger().info("2. Lowering directly onto block...")
        self.send_arm_trajectory(contact_state, duration_sec=2.5)
        time.sleep(3.0)

        # 3. Attach Object
        self.get_logger().info("3. Locking attach mechanism...")
        self.attach_pub.publish(Empty())
        time.sleep(1.0)

        # 4. Retract Upward
        self.get_logger().info("4. Retracting upward with block...")
        self.send_arm_trajectory(hover_state, duration_sec=2.5)
        time.sleep(3.0)

        self.get_logger().info("SUCCESS: MoveIt pick complete!")

def main():
    rclpy.init()
    node = MoveItPickPlaceNode()
    node.execute_pick()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
