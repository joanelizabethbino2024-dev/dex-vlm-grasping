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
import csv
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
TRIAL_LOG_PATH = os.path.expanduser("~/dexproject/vlm_scripts/trial_log.csv")

# --- Ablation / baseline switches (env vars so trials can be scripted without
# editing code between runs -- see PUBLICATION_READINESS.md for the trial
# protocol these are meant to support) ---
#
# DEFORMATION_TRIGGER_MODE:
#   "derivative"     -- current behavior: absolute ceiling OR effort-derivative trigger
#   "absolute_only"  -- ablation: drop the derivative trigger, keep only the ceiling
DEFORMATION_TRIGGER_MODE = os.environ.get("DEFORMATION_TRIGGER_MODE", "derivative")

# MNDF_MEMORY_ENABLED: "0" disables the per-object memory lookup/save so every
# grasp re-runs the full squeeze-relax-lock search (ablation for "does memory help").
MNDF_MEMORY_ENABLED = os.environ.get("MNDF_MEMORY_ENABLED", "1") != "0"

# FIXED_FORCE_BASELINE: "1" replaces the adaptive squeeze-relax-lock state
# machine with a naive close-to-a-fixed-closure baseline, for the "how much
# better than fixed force" comparison. Deformation is still monitored (using
# whatever DEFORMATION_TRIGGER_MODE is set) and logged, but never acted on --
# this baseline does not react to it.
FIXED_FORCE_BASELINE = os.environ.get("FIXED_FORCE_BASELINE", "0") == "1"
FIXED_FORCE_CLOSURE = float(os.environ.get("FIXED_FORCE_CLOSURE", "0.6"))

CLOSING_JOINTS = [
    "R_Thumb_Flexor", "R_Thumb_DIP",
    "R_Index_Flexor", "R_Index_DIP",
    "R_Middle_Flexor", "R_Middle_DIP",
    "R_Ring_Flexor", "R_Ring_DIP",
    "R_Pinky_Flexor", "R_Pinky_DIP",
]

# Real per-joint limits from expanded_robot.urdf <limit> tags (confirmed
# 2026-09-08, same values used in manual_target_node.py). Covers ALL 21 hand
# joints, not just CLOSING_JOINTS -- finger_joint_targets (from the VLM /
# detector) are documented everywhere else in this project as 0.0-1.0
# normalized values (0.5 = neutral for Yaw/Roll), so the "static" Pitch/
# Yaw/Roll targets need the same normalized->radians conversion as the
# closing joints get. Previously only CLOSING_JOINTS had limits, so
# publish_trajectory() fell back to treating static joint values as raw
# radians -- a normalized "0.5 neutral" Yaw was sent as 0.5 RADIANS, well
# past that joint's real +/-0.349 rad limit, driving it to a hard stop
# instead of neutral.
JOINT_LIMITS = {
    "R_Thumb_Pitch": (0.0, 1.047198), "R_Thumb_Roll": (-0.349066, 0.349066),
    "R_Thumb_Yaw": (-0.523599, 0.523599), "R_Thumb_Flexor": (0.0, 1.047198),
    "R_Thumb_DIP": (0.0, 1.047198),
    "R_Index_Pitch": (0.0, 1.308997), "R_Index_Yaw": (-0.349066, 0.349066),
    "R_Index_Flexor": (0.0, 1.047198), "R_Index_DIP": (0.0, 1.047198),
    "R_Middle_Pitch": (0.0, 1.308997), "R_Middle_Yaw": (-0.349066, 0.349066),
    "R_Middle_Flexor": (0.0, 1.047198), "R_Middle_DIP": (0.0, 1.047198),
    "R_Ring_Pitch": (0.0, 1.308997), "R_Ring_Yaw": (-0.349066, 0.349066),
    "R_Ring_Flexor": (0.0, 1.047198), "R_Ring_DIP": (0.0, 1.047198),
    "R_Pinky_Pitch": (0.0, 1.308997), "R_Pinky_Yaw": (-0.349066, 0.349066),
    "R_Pinky_Flexor": (0.0, 1.047198), "R_Pinky_DIP": (0.0, 1.047198),
}

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
    # RE-REVERTED (2026-09-16): briefly "fixed" this to the straightforward
    # lo+value*(hi-lo) form after a live /joint_states check seemed to show
    # hi=curled -- but that check was almost certainly catching the hand
    # mid-squeeze, not isolating cause and effect. manual_target_node.py's
    # _build_hand_positions (proven via actual live pick-and-lift testing,
    # not a rushed mid-session check) uses this exact hi-value*(hi-lo) form
    # with thumb_amount=0.0 explicitly labeled "thumb extended" -- i.e.
    # hi=open IS correct. Back to the original, trusted convention: for
    # this hand's actual joints, lo=fully curled/closed and hi=fully
    # open/extended.
    lo, hi = JOINT_LIMITS[joint]
    value = max(0.0, min(1.0, value))
    return hi - value * (hi - lo)


def radians_to_normalized(joint, value):
    # Inverse of normalized_to_radians's inverted convention (value = hi -
    # normalized*(hi-lo)  =>  normalized = (hi - value) / (hi - lo)).
    lo, hi = JOINT_LIMITS[joint]
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (hi - value) / (hi - lo)))


class AdaptiveGraspController(Node):
    def __init__(self):
        super().__init__("adaptive_grasp_controller")

        self.traj_pub = self.create_publisher(JointTrajectory, JOINT_TRAJECTORY_TOPIC, 10)
        self.create_subscription(String, FRAGILITY_TOPIC, self.on_fragility_msg, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.on_joint_states, 1)

        self.latest_effort = {}
        self.latest_position = {}
        self.effort_history = {name: deque(maxlen=EFFORT_WINDOW) for name in CLOSING_JOINTS}
        self.current_closure = {name: 0.0 for name in CLOSING_JOINTS}

        self.static_targets = {}
        self.object_name = "unknown object"
        self.grasp_active = False
        self.state = "idle"
        self.stable_count = 0
        self.mndf_memory = self.load_mndf_memory()

        # Per-trial bookkeeping for the CSV log (see PUBLICATION_READINESS.md).
        self.trial_start_time = None
        self.trial_deformation_events = 0
        self.trial_used_memory = False
        self._prev_deformed = False
        self._init_trial_log()

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_tick)
        mode_desc = "FIXED_FORCE_BASELINE" if FIXED_FORCE_BASELINE else f"adaptive/{DEFORMATION_TRIGGER_MODE}"
        self.get_logger().info(
            f"Adaptive grasp controller ready. mode={mode_desc} mndf_memory_enabled={MNDF_MEMORY_ENABLED}"
        )

    def _init_trial_log(self):
        if not os.path.exists(TRIAL_LOG_PATH):
            try:
                with open(TRIAL_LOG_PATH, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "timestamp", "object_name", "mode", "used_memory",
                        "time_to_lock_sec", "deformation_events",
                        "final_closure_mean", "final_effort_mean",
                    ])
            except Exception as e:
                self.get_logger().error(f"Failed to init trial log: {e}")

    def log_trial(self, time_to_lock_sec):
        final_effort_mean = 0.0
        efforts = [self.latest_effort.get(j, 0.0) for j in CLOSING_JOINTS if j in self.latest_effort]
        if efforts:
            final_effort_mean = sum(efforts) / len(efforts)
        final_closure_mean = sum(self.current_closure.values()) / len(self.current_closure)

        mode = "fixed_force" if FIXED_FORCE_BASELINE else DEFORMATION_TRIGGER_MODE
        try:
            with open(TRIAL_LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.time(), self.object_name, mode, self.trial_used_memory,
                    f"{time_to_lock_sec:.3f}", self.trial_deformation_events,
                    f"{final_closure_mean:.4f}", f"{final_effort_mean:.4f}",
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to write trial log: {e}")

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

        if MNDF_MEMORY_ENABLED and obj in self.mndf_memory:
            record = self.mndf_memory[obj]
            self.get_logger().info(f"Known object '{obj}' -- loading saved MNDF grasp.")
            self.current_closure = dict(record["closure"])
            self.static_targets = dict(analysis.get("finger_joint_targets", {}))
            self.publish_trajectory()
            self.state = "locked"
            self.grasp_active = True
            self.object_name = obj
            self.trial_used_memory = True
            self.log_trial(time_to_lock_sec=0.0)
            return

        if self.grasp_active and obj == self.object_name and self.state != "idle":
            targets = analysis.get("finger_joint_targets", {})
            self.static_targets = {k: v for k, v in targets.items() if k not in CLOSING_JOINTS}
            return

        self.object_name = obj
        self.trial_start_time = time.time()
        self.trial_deformation_events = 0
        self.trial_used_memory = False
        self._prev_deformed = False
        targets = analysis.get("finger_joint_targets", {})
        self.static_targets = {k: v for k, v in targets.items() if k not in CLOSING_JOINTS}
        # Seed from the hand's ACTUAL current position, not a blind 0.0 (fully
        # open). Confirmed live: if the hand is already partway closed (e.g.
        # pre-curled by a positioning node) when a "new object" arrives,
        # resetting to 0.0 snaps every closing joint open in one control tick
        # -- a sudden reversal that can slam a finger into the object/table
        # and knock it away (observed: an apple flung ~2.7m by this exact
        # snap-open on handoff) rather than a real deformation signal.
        self.current_closure = {
            name: radians_to_normalized(name, self.latest_position.get(name, JOINT_LIMITS[name][0]))
            for name in CLOSING_JOINTS
        }
        self.stable_count = 0
        self.grasp_active = True
        if FIXED_FORCE_BASELINE:
            self.state = "baseline_closing"
            self.get_logger().info(f"New object '{obj}' -- baseline: closing to fixed closure {FIXED_FORCE_CLOSURE}.")
        else:
            self.state = "squeezing"
            self.get_logger().info(f"New object '{obj}' -- starting adaptive squeeze from current closure {self.current_closure}.")

    def on_joint_states(self, msg: JointState):
        for name, position, effort in zip(msg.name, msg.position, msg.effort):
            if name in CLOSING_JOINTS:
                self.latest_position[name] = position
                if effort == effort:
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
            if abs(effort) > EFFORT_ABS_CEILING:
                return True, joint, effort, deriv
            if DEFORMATION_TRIGGER_MODE == "derivative" and deriv > EFFORT_DERIV_THRESHOLD:
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
        if deformed and not self._prev_deformed:
            self.trial_deformation_events += 1
        self._prev_deformed = deformed

        if self.state == "baseline_closing":
            # Naive fixed-force baseline: close straight to FIXED_FORCE_CLOSURE
            # and hold. Deformation is still detected/counted above (for the
            # "does fixed force damage fragile objects" comparison) but never
            # acted on -- that's the point of the baseline.
            for j in CLOSING_JOINTS:
                self.current_closure[j] = FIXED_FORCE_CLOSURE
            if self.all_efforts_stable():
                self.stable_count += 1
            else:
                self.stable_count = 0
            if self.stable_count >= STABLE_SAMPLES_TO_LOCK:
                self.lock_grasp()

        elif self.state == "squeezing":
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
        time_to_lock = (time.time() - self.trial_start_time) if self.trial_start_time else 0.0
        self.get_logger().info(
            f"Grasp locked for '{self.object_name}' at closure {self.current_closure} "
            f"after {time_to_lock:.1f}s, {self.trial_deformation_events} deformation event(s)."
        )
        self.log_trial(time_to_lock_sec=time_to_lock)

        if MNDF_MEMORY_ENABLED:
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
            normalized_to_radians(name, val) if name in JOINT_LIMITS
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
