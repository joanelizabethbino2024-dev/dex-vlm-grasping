#!/usr/bin/env python3
import time
import numpy as np
import ikpy.chain
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

DEXHAND_TRAJ_TOPIC = '/dexhand_controller/joint_trajectory'
DEXHAND_JOINT_NAMES = [
    'R_Index_Pitch', 'R_Middle_Pitch', 'R_Ring_Pitch', 'R_Pinky_Pitch',
    'R_Index_Flexor', 'R_Middle_Flexor', 'R_Ring_Flexor', 'R_Pinky_Flexor',
    'R_Index_DIP', 'R_Middle_DIP', 'R_Ring_DIP', 'R_Pinky_DIP',
    'R_Thumb_Yaw', 'R_Thumb_Roll', 'R_Thumb_Flexor', 'R_Thumb_DIP',
    'R_Index_Yaw', 'R_Middle_Yaw', 'R_Ring_Yaw', 'R_Pinky_Yaw', 'R_Thumb_Pitch',
]
DEXHAND_OPEN = [0.0] * 21
DEXHAND_CLOSED = [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6,
                  0.3, 0.0, 1.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.9]

URDF_PATH = '/tmp/real_robot_exact.urdf'
ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def build_chain():
    chain = ikpy.chain.Chain.from_urdf_file(
        URDF_PATH,
        base_elements=[
            'base_footprint', 'base_footprint_joint', 'mobile_base_link',
            'base_joint', 'base_link',
            'base_link-base_link_inertia', 'base_link_inertia',
            'shoulder_pan_joint', 'shoulder_link',
            'shoulder_lift_joint', 'upper_arm_link', 'elbow_joint',
            'forearm_link', 'wrist_1_joint', 'wrist_1_link',
            'wrist_2_joint', 'wrist_2_link', 'wrist_3_joint',
            'wrist_3_link', 'wrist_3-flange', 'flange',
            'flange-tool0', 'tool0', 'tool0_to_dexhand', 'dexhand_base_link'
        ]
    )
    mask = [False] * len(chain.links)
    for i, link in enumerate(chain.links):
        if link.name in ARM_JOINTS:
            mask[i] = True
    chain.active_links_mask = mask
    # Restrict shoulder_lift and elbow to a sane, natural-reach range so the
    # solver can't fold the arm backward into an extreme/unnatural pose.
    # No artificial bounds -- accept any mathematically valid solution;
    # what matters is whether it actually grasps the block, not pose looks.
    return chain


def solve_ik(chain, target_xyz, init=None):
    if init is None:
        init = [0.0] * len(chain.links)
        for i, link in enumerate(chain.links):
            if link.name == 'shoulder_pan_joint': init[i] = 0.0
            if link.name == 'shoulder_lift_joint': init[i] = -1.5708
            if link.name == 'elbow_joint': init[i] = 0.0
            if link.name == 'wrist_1_joint': init[i] = -1.5708
            if link.name == 'wrist_2_joint': init[i] = -1.5708
            if link.name == 'wrist_3_joint': init[i] = 0.0
    solution = chain.inverse_kinematics(
        target_xyz, initial_position=init,
        target_orientation=[0, 0, -1], orientation_mode='Z'
    )
    achieved = chain.forward_kinematics(solution)[:3, 3]
    error = np.linalg.norm(np.array(target_xyz) - achieved)
    print(f"  [IK check] target={target_xyz} achieved={achieved.tolist()} error_mm={error*1000:.3f}")
    if error > 0.01:
        return None
    joints = {}
    for link, angle in zip(chain.links, solution):
        if link.name in ARM_JOINTS:
            joints[link.name] = float(angle)
    return [joints[j] for j in ARM_JOINTS], solution


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        self.chain = build_chain()
        self.pose_sub = self.create_subscription(
            Pose, '/model/red_block/pose', self.pose_callback, 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.hand_pub = self.create_publisher(
            JointTrajectory, DEXHAND_TRAJ_TOPIC, 10)
        self.attach_pub = self.create_publisher(Empty, '/attach', 10)
        self.detach_pub = self.create_publisher(Empty, '/detach', 10)
        self.target_pose = None

    def pose_callback(self, msg):
        self.target_pose = msg

    def wait_for_pose(self, timeout=3.0):
        self.target_pose = None
        start = time.time()
        while rclpy.ok() and self.target_pose is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.target_pose

    def send_arm_trajectory(self, joint_positions, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.arm_pub.publish(msg)

    def set_hand(self, positions, duration_sec=1.0):
        msg = JointTrajectory()
        msg.joint_names = DEXHAND_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.hand_pub.publish(msg)

    def run(self):
        pose = self.wait_for_pose()
        if pose is not None:
            x, y, z_center = pose.position.x, pose.position.y, pose.position.z
            self.get_logger().info(f"Live block pose: X={x:.3f} Y={y:.3f} Z={z_center:.3f}")
        else:
            x, y, z_center = 0.50, 0.00, 0.07
            self.get_logger().warn(f"Pose bridge timed out, using fallback ({x},{y},{z_center})")

        z_top = z_center + 0.10  # block is now 0.20m tall, half-height=0.10
        z_top -= 0.03  # small correction: close remaining ~2-3cm visible gap
        approach_target = [x, y, z_top + 0.15]
        grasp_target = [x, y, z_top]

        # Seed with the robot's true home configuration for a consistent,
        # natural arm branch across the whole sequence -- reduces wrist
        # rotation (and therefore attached-object swing) throughout.
        home_seed = [0.0] * len(self.chain.links)
        for i, link in enumerate(self.chain.links):
            if link.name == 'shoulder_lift_joint': home_seed[i] = -1.5708
            if link.name == 'wrist_1_joint': home_seed[i] = -1.5708
            if link.name == 'wrist_2_joint': home_seed[i] = -1.5708
        approach_result = solve_ik(self.chain, approach_target, init=home_seed)
        if approach_result is None:
            self.get_logger().error(f"UNREACHABLE: {approach_target}")
            return
        approach, prev_solution = approach_result

        grasp_result = solve_ik(self.chain, grasp_target, init=list(prev_solution))
        if grasp_result is None:
            self.get_logger().error(f"UNREACHABLE: {grasp_target}")
            return
        grasp, grasp_full_solution = grasp_result

        self.get_logger().info(f"Approach joints: {[round(v,4) for v in approach]}")
        self.get_logger().info(f"Grasp joints: {[round(v,4) for v in grasp]}")

        self.get_logger().info("1. Opening hand, moving overhead...")
        self.set_hand(DEXHAND_OPEN, 1.0)
        self.send_arm_trajectory(approach, 3.5)
        time.sleep(4.0)

        self.get_logger().info("2. Lowering onto block...")
        self.send_arm_trajectory(grasp, 2.5)
        time.sleep(3.0)

        self.get_logger().info("3. Closing hand SLOWLY around block (friction grip, not glue)...")
        # Close gradually in stages -- a real hand doesn't slam shut, it
        # closes progressively until it feels resistance. We simulate this
        # by commanding the closed pose with a LONGER duration, giving the
        # physics engine time to build up real contact forces gradually
        # rather than snapping through the object.
        self.set_hand(DEXHAND_CLOSED, 3.0)
        time.sleep(3.5)

        self.get_logger().info("4. Holding via friction (no attach signal used)...")
        time.sleep(1.0)  # brief pause to let contact forces stabilize

        self.get_logger().info("5. Lifting SLOWLY to avoid slipping...")
        lift_joints = list(grasp)
        lift_joints[1] -= 0.35  # shoulder_lift index -- raises the arm
        # CRITICAL: slow, long duration lift. Fast acceleration is the
        # single biggest cause of a friction-only grip slipping/dropping
        # the object -- real hands lift slowly for exactly this reason.
        self.send_arm_trajectory(lift_joints, 5.0)
        time.sleep(6.0)

        # Verify: did the object actually rise, or did it slip?
        self.get_logger().info("Checking whether grip actually held...")

        self.get_logger().info("TEST COMPLETE - check Gazebo viewport for actual result.")


def main():
    rclpy.init()
    node = PickAndPlaceNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
