#!/usr/bin/env python3
"""
Adaptive Squeeze-and-Relax Grasp Controller
--------------------------------------------
Novel behavior: ramps finger closure up under a firm initial force target,
continuously watches joint EFFORT (not just position) for the onset of
deformation using a rate-of-change (derivative) trigger, backs off the
instant deformation is detected, then fine-approaches back in and LOCKS the
grasp at that self-calibrated limit. That limit is logged per object as a
"Maximum Non-Deforming Force" (MNDF) record so future grasps of the same
object skip the guessing phase entirely (Experience Layer).

No dedicated tactile/F-T hardware required -- this runs purely off
/joint_states effort, which ros2_control / gazebo_ros2_control already
publishes for every actuated joint.
"""
import json
import os
import time
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_TRAJECTORY_TOPIC = "/dexhand_controller/joint_trajectory"
FRAGILITY_TOPIC = "/gripper_camera/fragility_analysis"
JOINT_STATES_TOPIC = "/joint_states"
MNDF_LOG_PATH = os.path.expanduser("~/dexproject/vlm_scripts/mndf_memory.json")

CLOSING_JOINTS = [
    "R_Thumb_Flexor", "R_Thumb_DIP",
    "R_Index_Flexor", "R_Index_DIP",
    "R_Middle_Flexor", "R_Middle_DIP",
    "R_Ring_Flexor", "R_Ring_DIP",
    "R_Pinky_Flexor", "R_Pinky_DIP",
]

# Real limits confirmed from dexhandv2_right.xacro: lower=0.0 upper=1.047198 (60deg)
JOINT_LIMITS = {name: (0.0, 1.047198) for name in CLOSING_JOINTS}

# dexhand_controller (merged_controllers.yaml) manages ONLY the 21 hand joints,
# separately from the arm's joint_trajectory_controller, and has
# allow_partial_joints_goal: true -- so we do NOT need to include arm joints
# in trajectories sent here.

CONTROL_RATE_HZ = 20.0
RAMP_STEP = 0.02
RELAX_STEP = 0.05
FINE_APPROACH_STEP = 0.005
EFFORT_WINDOW = 5
EFFORT_DERIV_THRESHOLD = 0.15
EFFORT_ABS_CEILING = 2.0
STABLE_SAMPLES_TO_LOCK = 15


def normalized_to_radians(joint, value):
    lo, hi = JOINT_LIMITS[joint]
    value = max(0.0, min(1.0, value))
    return lo + value * (hi - lo)


class AdaptiveGraspController(Node):
    def __init__(self):
        super().__init__("adaptive_grasp_controller")

        self.traj_pub = self.create_publisher(JointTrajectory, JOINT_TRAJECTORY_TOPIC, 10)
        self.create_subscription(String, FRAGILITY_TOPIC, self.on_fragility_msg, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.on_joint_states, 1)

        self.latest_effort = {}
        self.effort_history = {name: deque(maxlen=EFFORT_WINDOW) for name in CLOSING_JOINTS}
        self.current_closure = {name: 0.0 for name in CLOSING_JOINTS}

        self.static_targets = {}
        self.object_name = "unknown object"
        self.grasp_active = False
        self.state = "idle"
        self.stable_count = 0
        self.mndf_memory = self.load_mndf_memory()

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_tick)
        self.get_logger().info("Adaptive grasp controller ready.")

    def load_mndf_memory(self):
        if os.path.exists(MNDF_LOG_PATH):
            try:
                with open(MNDF_LOG_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_mndf_memory(self):
        try:
            with open(MNDF_LOG_PATH, "w") as f:
                json.dump(self.mndf_memory, f, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to write MNDF memory: {e}")

    def on_fragility_msg(self, msg: String):
        try:
            analysis = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        obj = analysis.get("object_name", "unknown object")
        if obj == "none visible":
            self.grasp_active = False
            self.state = "idle"
            return

        if obj in self.mndf_memory:
            record = self.mndf_memory[obj]
            self.get_logger().info(f"Known object '{obj}' -- loading saved MNDF grasp.")
            self.current_closure = dict(record["closure"])
            self.static_targets = dict(analysis.get("finger_joint_targets", {}))
            self.publish_trajectory()
            self.state = "locked"
            self.grasp_active = True
            self.object_name = obj
            return

        if self.grasp_active and obj == self.object_name and self.state != "idle":
            targets = analysis.get("finger_joint_targets", {})
            self.static_targets = {k: v for k, v in targets.items() if k not in CLOSING_JOINTS}
            return

        self.object_name = obj
        targets = analysis.get("finger_joint_targets", {})
        self.static_targets = {k: v for k, v in targets.items() if k not in CLOSING_JOINTS}
        self.current_closure = {name: 0.0 for name in CLOSING_JOINTS}
        self.stable_count = 0
        self.state = "squeezing"
        self.grasp_active = True
        self.get_logger().info(f"New object '{obj}' -- starting adaptive squeeze.")

    def on_joint_states(self, msg: JointState):
        for name, effort in zip(msg.name, msg.effort):
            if name in CLOSING_JOINTS and effort == effort:
                self.latest_effort[name] = effort
                self.effort_history[name].append(effort)

    def effort_derivative(self, joint):
        hist = self.effort_history[joint]
        if len(hist) < 2:
            return 0.0
        return abs(hist[-1] - hist[0]) / max(1, len(hist))

    def deformation_detected(self):
        for joint in CLOSING_JOINTS:
            effort = self.latest_effort.get(joint, 0.0)
            deriv = self.effort_derivative(joint)
            if abs(effort) > EFFORT_ABS_CEILING or deriv > EFFORT_DERIV_THRESHOLD:
                return True, joint, effort, deriv
        return False, None, 0.0, 0.0

    def all_efforts_stable(self):
        for joint in CLOSING_JOINTS:
            if self.effort_derivative(joint) > (EFFORT_DERIV_THRESHOLD * 0.3):
                return False
        return True

    def control_tick(self):
        if not self.grasp_active or self.state in ("idle", "locked"):
            return

        deformed, joint, effort, deriv = self.deformation_detected()

        if self.state == "squeezing":
            if deformed:
                self.get_logger().warning(
                    f"Deformation onset on {joint} (effort={effort:.3f}, "
                    f"d_effort={deriv:.3f}) -- relaxing."
                )
                self.state = "relaxing"
                self.stable_count = 0
            else:
                for j in CLOSING_JOINTS:
                    self.current_closure[j] = min(1.0, self.current_closure[j] + RAMP_STEP)

        elif self.state == "relaxing":
            for j in CLOSING_JOINTS:
                self.current_closure[j] = max(0.0, self.current_closure[j] - RELAX_STEP)
            if not deformed:
                self.state = "fine_approach"
                self.stable_count = 0

        elif self.state == "fine_approach":
            if deformed:
                self.state = "relaxing"
                self.stable_count = 0
            else:
                for j in CLOSING_JOINTS:
                    self.current_closure[j] = min(1.0, self.current_closure[j] + FINE_APPROACH_STEP)
                if self.all_efforts_stable():
                    self.stable_count += 1
                else:
                    self.stable_count = 0
                if self.stable_count >= STABLE_SAMPLES_TO_LOCK:
                    self.lock_grasp()

        self.publish_trajectory()

    def lock_grasp(self):
        self.state = "locked"
        self.get_logger().info(
            f"Grasp locked for '{self.object_name}' at closure "
            f"{self.current_closure}. Saving to MNDF memory."
        )
        self.mndf_memory[self.object_name] = {
            "closure": dict(self.current_closure),
            "timestamp": time.time(),
        }
        self.save_mndf_memory()

    def publish_trajectory(self):
        traj = JointTrajectory()

        all_targets = dict(self.static_targets)
        for j in CLOSING_JOINTS:
            all_targets[j] = self.current_closure[j]

        traj.joint_names = list(all_targets.keys())
        point = JointTrajectoryPoint()
        point.positions = [
            normalized_to_radians(name, val) if name in CLOSING_JOINTS
            else float(val) if isinstance(val, (int, float)) else 0.0
            for name, val in all_targets.items()
        ]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(1.0 / CONTROL_RATE_HZ * 1e9)
        traj.points.append(point)
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveGraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
