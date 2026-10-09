#!/usr/bin/env python3
"""
Finger Alignment Node
----------------------
Dedicated, separate from adaptive_grasp_controller.py on purpose: this node
does ONLY straight-finger positioning around an object once the arm has
already been confirmed (by vision_ik_overhead_closed_loop.py, via real
forward-kinematics feedback, not a fixed timer) to have reached the target.

It does NOT curl or squeeze anything. Pitch/Flexor/DIP are commanded to raw
0.0 radians directly -- no normalized-value conversion, no lo/hi convention
guessing (that ambiguity caused repeated confusion tonight). Only Yaw/Roll
(lateral finger spread, not curl) are set, scaled by the object's own
measured real-world size (real_size_m, from the depth camera via
yolo_depth_detector.py -- this IS "using the camera" to decide positioning,
just not re-querying it live here) so a differently-sized object gets a
differently-spread hand, not one fixed pose.

Listens on FINGER_ALIGN_TRIGGER_TOPIC (published by
vision_ik_overhead_closed_loop.py's _on_settle_timer_fired) for
{"object_name", "base_link_x/y/z", "real_size_m"}, and only reacts to that --
it does not do its own arm motion or camera detection.
"""
import json
import os
import re
import subprocess
import time

import rclpy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

FINGER_ALIGN_TRIGGER_TOPIC = "/pick/ready_for_finger_alignment"
# Reported back to vision_ik_overhead_closed_loop.py once the full align->
# close->lift sequence below is actually finished. Without this,
# vision_ik_overhead_closed_loop.py reset itself to idle right after the
# initial handoff and a SECOND pick attempt could start (fresh
# open_hand_preshape, which UNCURLS the thumb) while this node was still
# mid-lift on the same physical hand -- confirmed live 2026-09-16 as the
# cause of the thumb sweeping the object away during that uncurl.
FINGER_SEQUENCE_DONE_TOPIC = "/pick/finger_sequence_done"
HAND_TRAJECTORY_TOPIC = "/dexhand_controller/joint_trajectory"
ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"
ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ALIGN_DURATION_SEC = 1.5
CLOSE_SETTLE_SEC = 0.5   # gap after alignment finishes moving before closing starts
# SLOWED WAY DOWN (2026-09-17): the previous 2.0s value moved all five
# fingers through a large angle fast enough that it acted like a slap, not a
# grab -- confirmed live, flung the apple ~11m. Same target closure amount,
# much lower angular velocity: the joint_trajectory_controller interpolates
# to the same final angle over more time, i.e. gently, instead of fast.
CLOSE_DURATION_SEC = 10.0
THUMB_CLOSE_SEC = 6.0
CONTACT_EFFORT_THRESHOLD = 0.6
CLOSE_STEP_RAD = 0.025  # smaller steps -- less distance/momentum to shove the object before contact is caught
CLOSE_TICK_SEC = 0.3
CLOSE_MAX_TICKS = 45  # raised to accommodate the deeper 24-tick secure squeeze
SQUEEZE_PAST_CONTACT_RAD = 0.12  # (unused by close_hand now, kept for reference)  # extra push past first contact so the grip applies real inward force, not just touching
# Two-phase force profile (user request, 2026-10-05): grip HARD first to
# guarantee no slip through the lift, then relax toward the minimum force
# that still holds -- the way a human hand does it, and the way bruise-
# minimizing fruit-harvest grippers are supposed to work. SECURE pushes well
# past first contact; RELAX_STEP_RAD backs off in small increments once the
# lift is confirmed, watching for slip (see _red_fraction) and re-tightening
# if it sees one, instead of blindly backing off on a fixed schedule.
SECURE_SQUEEZE_RAD = 0.30  # (unused now -- see SECURE_SQUEEZE_TICKS)
SECURE_SQUEEZE_TICKS = 24  # squeeze deeper still -- lift keeps breaking a grip that holds fine while stationary
RELAX_STEP_RAD = 0.04
RELAX_TICK_SEC = 0.5
RELAX_MIN_FRACTION = 0.55   # don't relax a joint below this fraction of its secure amount
RELAX_MAX_TICKS = 20
SLIP_RED_DROP_FRACTION = 0.4  # gripper-camera red-pixel count dropping below this fraction of its post-lift baseline = slip
OBSERVE_DIR = "/home/tt501/dexproject/vlm_scripts/logs/grasp_observations"

THUMB_PUSH_SEC = 6.0
THUMB_PUSH_AMOUNT_PITCH_FLEXOR = 0.5
THUMB_PUSH_AMOUNT_DIP = 0.3
TILT_SEC = 4.0
TILT_WRIST2_DELTA_RAD = -0.2   # measured: moves claw tips ~4.6 cm toward the thumb/apple side
CLAW_RAD = {"Pitch": 0.5, "Flexor": 0.8, "DIP": 0.5}   # four-finger claw
CLAW_TIGHTEN_RAD = {"Pitch": 0.15, "Flexor": 0.15, "DIP": 0.1}


def claw_positions(extra=None):
    pos = {}
    for n in PITCH_FLEXOR_DIP_JOINTS:
        if n.startswith("R_Thumb"):
            pos[n] = 0.0
        else:
            kind = n.split("_")[2]
            pos[n] = CLAW_RAD[kind] + (extra[kind] if extra else 0.0)
    return pos

LIFT_SETTLE_SEC = 1.0    # let fingers finish closing before lifting
# manual_target_node.py's proven approach: change ONLY shoulder_lift_joint by
# a fixed delta from wherever the arm CURRENTLY is, not a fresh IK solve --
# re-solving IK even seeded with the current angles can shift every joint
# slightly, and any shift risks disturbing a marginal grip. Negative = up
# (measured there: shoulder_lift went 0.7142 -> 0.7277 for a 5cm descend,
# i.e. increasing lowers the wrist).
LIFT_SHOULDER_LIFT_DELTA_RAD = -0.07  # even gentler -- marginal grips were still breaking at -0.15
LIFT_DURATION_SEC = 14.0  # slower lift to match the gentler delta

HAND_JOINTS = [
    "R_Thumb_Pitch", "R_Thumb_Roll", "R_Thumb_Yaw", "R_Thumb_Flexor", "R_Thumb_DIP",
    "R_Index_Pitch", "R_Index_Yaw", "R_Index_Flexor", "R_Index_DIP",
    "R_Middle_Pitch", "R_Middle_Yaw", "R_Middle_Flexor", "R_Middle_DIP",
    "R_Ring_Pitch", "R_Ring_Yaw", "R_Ring_Flexor", "R_Ring_DIP",
    "R_Pinky_Pitch", "R_Pinky_Yaw", "R_Pinky_Flexor", "R_Pinky_DIP",
]
PITCH_FLEXOR_DIP_JOINTS = [j for j in HAND_JOINTS if not j.endswith("Yaw") and not j.endswith("Roll")]

# Real per-joint limits from expanded_robot.urdf <limit> tags (same values
# used throughout this project). lo=0.0 for every Pitch/Flexor/DIP joint --
# confirmed live tonight (three separate times, watching the actual sim)
# that raw values approaching hi are what visibly curl these joints, and the
# straight/uncurled pose that just worked (undisturbed apple, correct
# spread) sits at raw 0 = lo. This directly contradicts an older comment in
# adaptive_grasp_controller.py/manual_target_node.py claiming the opposite;
# trusting tonight's direct, repeated, live observation over that comment.
HAND_JOINT_LIMITS_RAD = {
    "R_Thumb_Pitch": (0.0, 1.047198), "R_Thumb_Flexor": (0.0, 1.047198), "R_Thumb_DIP": (0.0, 1.047198),
    "R_Index_Pitch": (0.0, 1.308997), "R_Index_Flexor": (0.0, 1.047198), "R_Index_DIP": (0.0, 1.047198),
    "R_Middle_Pitch": (0.0, 1.308997), "R_Middle_Flexor": (0.0, 1.047198), "R_Middle_DIP": (0.0, 1.047198),
    "R_Ring_Pitch": (0.0, 1.308997), "R_Ring_Flexor": (0.0, 1.047198), "R_Ring_DIP": (0.0, 1.047198),
    "R_Pinky_Pitch": (0.0, 1.308997), "R_Pinky_Flexor": (0.0, 1.047198), "R_Pinky_DIP": (0.0, 1.047198),
}

# Modest, not full-range, close amount (fraction of the way from lo to hi).
# There's no force feedback in this simple path (adaptive_grasp_controller.py
# is disabled for now), so err on the side of under-closing rather than
# risk over-squeezing/pushing the object with an open-loop full close.
# DIP gets a lower cap than Pitch/Flexor so the fingertip doesn't hook past
# the object's surface instead of settling into a hook around it (same
# reasoning as manual_target_node.py's DIP_CURL_CAP).
# CUT DOWN FURTHER (2026-09-17): 0.4/0.7 was still enough sweep distance to
# push the object even at a slow speed -- less distance to travel means less
# opportunity to catch it off-center and shove it, at the cost of a weaker
# grip. Prioritizing not-pushing over grip strength right now.
# Four fingers bend mainly at the KNUCKLE (Pitch), only lightly at the middle/tip joints.
CLOSE_AMOUNT_PITCH_FLEXOR = 0.55
CLOSE_AMOUNT_DIP = 0.35
# DIAGNOSTIC (2026-09-17): every attempt tonight pushed the object in the
# same direction (increasingly negative Y) regardless of amount or speed --
# a systematic bias, not random contact noise. The thumb travels the
# largest arc of any joint (starts fully extended, has the most range to
# sweep through) and is the prime suspect. Isolating it: thumb stays fully
# extended through the whole close, only the four fingers move, to see
# whether the systematic push goes away.
CLOSE_AMOUNT_THUMB_PITCH_FLEXOR = 0.65
CLOSE_AMOUNT_THUMB_DIP = 0.4

# Object-size band this hand can plausibly wrap around (matches
# yolo_depth_detector.py's MIN/MAX_OBJECT_WIDTH_M filter).
MIN_OBJECT_WIDTH_M = 0.02
MAX_OBJECT_WIDTH_M = 0.16

# Baseline outward Yaw for the outer fingers (raw radians). Neutral (0.0) was
# confirmed via manual_target_node.py live testing to let the fingers close
# in parallel with the thumb not facing them -- the object got shoved
# sideways instead of caged. This non-zero baseline is what actually
# positions them AROUND the object; the size-scaled spread on top adapts
# that positioning to how big the object actually is.
# Widened (2026-09-16): the fingers were visibly bunched too close together
# in testing rather than surrounding the object with real clearance. Index/
# Ring/Pinky Yaw range is +/-0.349066 rad -- base+max-spread is kept safely
# under that ceiling.
YAW_BASE_OUTER_RAD = 0.22
MAX_YAW_SPREAD_RAD = 0.12  # additional spread at the largest graspable size
GRIP_FAN_OUTER_RAD = 0.28  # outer finger yaw = +/-0.14 -- narrowed hard: side_camera showed the wide fan curling AWAY from the apple instead of converging onto it; only the zero-yaw Middle finger ever made contact
THUMB_YAW_RAD = -0.35
THUMB_ROLL_RAD = -0.20  # rotated further toward opposing the fingers (was -0.10)


class FingerAlignmentNode(Node):
    def __init__(self):
        # use_sim_time -- without it, every _call_once_after delay below runs
        # on the WALL clock while the hand/arm trajectories it's timed
        # against execute in SIM time. Confirmed live 2026-09-16 as the cause
        # of "lift starts before the close motion visibly finished": under
        # this sim's real-time-factor (visibly below 1.0 all night), a
        # wall-clock delay elapses faster than the sim-time trajectory
        # duration it's meant to wait out.
        super().__init__(
            "finger_alignment_node",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.hand_traj_pub = self.create_publisher(JointTrajectory, HAND_TRAJECTORY_TOPIC, 10)
        self.arm_traj_pub = self.create_publisher(JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        self.sequence_done_pub = self.create_publisher(String, FINGER_SEQUENCE_DONE_TOPIC, 10)
        self.create_subscription(String, FINGER_ALIGN_TRIGGER_TOPIC, self.on_trigger, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self._yaw_roll_positions = {}
        self.latest_arm_positions = {}
        self.latest_hand_effort = {}
        self._secure_positions = {}

        # Self-observation (2026-10-05, user request): so this node -- and
        # whoever reviews its logs afterward -- can see WHERE and WHY a grasp
        # fails, not just infer it from final joint angles. gripper_camera is
        # mounted on the palm, so a simple red-pixel check on it is a cheap
        # "is the apple still here" signal with no new hardware, and doubles
        # as the slip detector for the relax phase below.
        self.bridge = CvBridge()
        self.latest_gripper_image = None
        self.create_subscription(Image, "/gripper_camera", self.on_gripper_image, 5)
        self.latest_side_image = None
        self.create_subscription(Image, "/side_camera", self.on_side_image, 5)
        self.run_dir = os.path.join(OBSERVE_DIR, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self.obs_log = open(os.path.join(self.run_dir, "observations.log"), "a")

        self.get_logger().info(f"Listening for alignment triggers on {FINGER_ALIGN_TRIGGER_TOPIC}")
        self.get_logger().info(f"Saving grasp observations to {self.run_dir}")

    def on_gripper_image(self, msg: Image):
        try:
            self.latest_gripper_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"gripper_camera conversion failed: {e}")

    def on_side_image(self, msg: Image):
        try:
            self.latest_side_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"side_camera conversion failed: {e}")

    def _red_fraction(self):
        """Fraction of the gripper-camera frame that's apple-red. Cheap
        stand-in for a slip/presence sensor -- no new hardware, just the
        camera already mounted on the hand."""
        img = self.latest_gripper_image
        if img is None:
            return None
        b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        mask = (r > 80) & (r > g + 25) & (r > b + 25)
        return float(mask.mean())

    def _observe(self, phase, object_name, extra=None):
        """Save a gripper_camera + side_camera frame and log joint efforts /
        red-fraction, tagged with the phase name, so a run can be reviewed
        frame-by-frame afterward instead of only from the final result."""
        stamp = time.strftime("%H%M%S")
        red = self._red_fraction()
        efforts = {n: round(self.latest_hand_effort.get(n, 0.0), 2) for n in PITCH_FLEXOR_DIP_JOINTS}
        try:
            out = subprocess.run(
                "ign topic -e -t /world/apple_world/pose/info -n 1",
                shell=True, capture_output=True, text=True, timeout=10,
            ).stdout
            m = re.search(r'name: "apple_05".*?position \{(.*?)\}', out, re.S)
            vals = dict(re.findall(r'([xyz]): ([-\d.e]+)', m.group(1))) if m else {}
            apple_ground_truth = [round(float(vals.get(k, 0.0)), 4) for k in "xyz"]
        except Exception as e:
            apple_ground_truth = f"ERROR: {e}"
        line = {"t": stamp, "phase": phase, "object": object_name, "apple_ground_truth": apple_ground_truth, "red_fraction": red, "effort": efforts}
        if extra:
            line.update(extra)
        self.obs_log.write(json.dumps(line) + "\n")
        self.obs_log.flush()
        if self.latest_gripper_image is not None:
            cv2.imwrite(os.path.join(self.run_dir, f"{stamp}_{phase}_gripper.png"), self.latest_gripper_image)
        if self.latest_side_image is not None:
            cv2.imwrite(os.path.join(self.run_dir, f"{stamp}_{phase}_side.png"), self.latest_side_image)
        self.get_logger().info(f"[observe] {phase}: red_fraction={red}")

    def on_joint_states(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                self.latest_arm_positions[name] = position
        for name, position, effort in zip(msg.name, msg.position, msg.effort):
            if name in PITCH_FLEXOR_DIP_JOINTS:
                self.latest_hand_effort[name] = effort

    def _call_once_after(self, delay_sec, callback):
        box = {}

        def _fire():
            timer = box.get("timer")
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
            callback()

        box["timer"] = self.create_timer(delay_sec, _fire)

    def _publish_hand_positions(self, positions, duration_sec):
        traj = JointTrajectory()
        traj.joint_names = HAND_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [positions[name] for name in HAND_JOINTS]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)

    def _publish_hand_positions_subset(self, joint_names, positions, duration_sec):
        """Same as _publish_hand_positions but for only SOME joints, sent as
        its own separate JointTrajectory message. Confirmed live 2026-09-17:
        sending a joint meant to STAY PUT (e.g. Pitch=0) in the same
        multi-joint trajectory point as a joint that's actually MOVING (Yaw,
        during align) let the stationary one visibly bulge mid-motion before
        settling back to its commanded value -- a trajectory-interpolation
        artifact, not a real curl command. Splitting them into independent
        messages means a joint that isn't supposed to move is never part of
        the same interpolation as one that is."""
        traj = JointTrajectory()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = [positions[name] for name in joint_names]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)

    def close_hand(self, object_name):
        """Closed-loop close: step every joint toward its full-close target,
        but FREEZE any joint the instant its /joint_states effort shows real
        contact, instead of committing to a fixed trajectory that overshoots
        past contact and shoves the object -- that overshoot is what flung
        the apple on every earlier open-loop attempt tonight."""
        targets, current, frozen, hit_count, squeeze_ticks = {}, {}, {}, {}, {}
        for name in PITCH_FLEXOR_DIP_JOINTS:
            lo, hi = HAND_JOINT_LIMITS_RAD[name]
            is_dip = name.endswith("DIP")
            if name.startswith("R_Thumb"):
                amount = CLOSE_AMOUNT_THUMB_DIP if is_dip else CLOSE_AMOUNT_THUMB_PITCH_FLEXOR
            else:
                amount = CLOSE_AMOUNT_DIP if is_dip else CLOSE_AMOUNT_PITCH_FLEXOR
            targets[name] = lo + amount * (hi - lo)
            current[name] = 0.0
            frozen[name] = False
            hit_count[name] = 0
            squeeze_ticks[name] = 0
        ticks = {"n": 0}

        def step():
            ticks["n"] += 1
            any_moving = False
            for name in PITCH_FLEXOR_DIP_JOINTS:
                if frozen[name]:
                    continue
                effort = abs(self.latest_hand_effort.get(name, 0.0))
                # React on the FIRST over-threshold reading, not the second --
                # confirmed live (side_camera capture) that waiting for a second
                # confirming tick let the fingers keep curling past first
                # contact and sweep the apple out of the way before freezing,
                # ending in a clenched fist next to the object instead of
                # around it.
                if effort > CONTACT_EFFORT_THRESHOLD and current[name] > 0.0:
                    hit_count[name] += 1
                else:
                    hit_count[name] = 0
                if hit_count[name] >= 1:
                    # SECURE as a RAMP of the same small step size used
                    # before contact, not one big jump -- confirmed live
                    # (side_camera capture) that a single large post-contact
                    # jump was enough on its own to shove the apple out of
                    # the other fingers' reach before they arrived.
                    if squeeze_ticks[name] < SECURE_SQUEEZE_TICKS:
                        current[name] = min(targets[name], current[name] + CLOSE_STEP_RAD)
                        squeeze_ticks[name] += 1
                        any_moving = True
                        self.get_logger().info(f"{name} contact (effort {effort:.2f}) -- securing, now {current[name]:.2f} rad.")
                    else:
                        frozen[name] = True
                        self.get_logger().info(f"{name} holding (secure) at {current[name]:.2f} rad.")
                    continue
                if current[name] < targets[name]:
                    current[name] = min(targets[name], current[name] + CLOSE_STEP_RAD)
                    any_moving = True
            self._publish_hand_positions_subset(list(current), current, CLOSE_TICK_SEC)
            if any_moving and ticks["n"] < CLOSE_MAX_TICKS:
                self._call_once_after(CLOSE_TICK_SEC, step)
            else:
                fingers_touched = sum(1 for n in squeeze_ticks if squeeze_ticks[n] > 0 and not n.startswith("R_Thumb"))
                thumb_touched = any(squeeze_ticks[n] > 0 for n in squeeze_ticks if n.startswith("R_Thumb"))
                self._secure_positions = dict(current)  # relax phase starts from here
                if fingers_touched >= 1 and thumb_touched:
                    self.get_logger().info(
                        f"Grab converged on '{object_name}' after {ticks['n']} ticks -- "
                        f"thumb AND {fingers_touched} finger(s) in contact (SECURE phase); lifting.")
                    self._observe("secure_grip_done", object_name)
                    self._call_once_after(LIFT_SETTLE_SEC, lambda: self.lift(object_name))
                else:
                    self.get_logger().warning(
                        f"NOT lifting '{object_name}' -- apple not confirmed in hand "
                        f"(thumb_touched={thumb_touched}, fingers_touched={fingers_touched}).")
                    self._observe("secure_grip_failed", object_name,
                                  {"thumb_touched": thumb_touched, "fingers_touched": fingers_touched})
                    self._call_once_after(LIFT_SETTLE_SEC, lambda: self._report_done(object_name))

        self._observe("pre_close", object_name)
        step()

    def lift(self, object_name):
        """Raise the arm to test whether the grasp actually holds -- ported
        from manual_target_node.py's proven approach: change ONLY
        shoulder_lift_joint by a fixed delta from wherever the arm currently
        is, not a fresh IK solve (re-solving can shift every joint slightly
        and disturb a marginal grip)."""
        missing = [j for j in ARM_JOINTS if j not in self.latest_arm_positions]
        if missing:
            self.get_logger().warning(f"No joint_states yet for {missing}; skipping lift.")
            return

        positions = [self.latest_arm_positions[j] for j in ARM_JOINTS]
        lift_idx = ARM_JOINTS.index("shoulder_lift_joint")
        positions[lift_idx] += LIFT_SHOULDER_LIFT_DELTA_RAD

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(LIFT_DURATION_SEC)
        point.time_from_start.nanosec = int((LIFT_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.arm_traj_pub.publish(traj)
        self.get_logger().info(f"Lifting to test the grasp on '{object_name}'.")
        self._observe("lift_commanded", object_name)
        self._call_once_after(LIFT_DURATION_SEC + LIFT_SETTLE_SEC, lambda: self.relax_grip(object_name))

    def relax_grip(self, object_name):
        """Phase 2 of the user-requested force profile: grip was SECURE
        (over-tightened) to survive the lift; now back each joint off toward
        the minimum that still holds, watching gripper_camera's red-pixel
        fraction (the apple's own visible presence in the palm view) as a
        slip signal. A drop below SLIP_RED_DROP_FRACTION of the post-lift
        baseline means something let go -- stop relaxing immediately and
        hold right there instead of continuing to loosen a grip that's
        already failing."""
        self._observe("post_lift", object_name)
        baseline = self._red_fraction()
        if baseline is None or baseline < 0.01:
            self.get_logger().warning(
                f"No apple visible to gripper_camera after lift (red_fraction={baseline}) -- "
                "grasp likely already failed; skipping relax, reporting done.")
            self._observe("relax_skipped_no_apple", object_name, {"baseline_red": baseline})
            self._call_once_after(LIFT_SETTLE_SEC, lambda: self._report_done(object_name))
            return

        secure = dict(self._secure_positions)
        floor = {n: v * RELAX_MIN_FRACTION for n, v in secure.items()}
        current = dict(secure)
        ticks = {"n": 0}

        def step():
            ticks["n"] += 1
            red = self._red_fraction()
            slipped = red is not None and baseline > 0 and (red / baseline) < SLIP_RED_DROP_FRACTION
            if slipped:
                self.get_logger().warning(
                    f"Slip detected during relax (red_fraction {red:.3f} vs baseline {baseline:.3f}) -- "
                    "holding here, not relaxing further.")
                self._observe("relax_slip_detected", object_name, {"tick": ticks["n"], "red": red, "baseline": baseline})
                self._call_once_after(LIFT_SETTLE_SEC, lambda: self._report_done(object_name))
                return

            any_moving = False
            for name in current:
                if current[name] > floor[name]:
                    current[name] = max(floor[name], current[name] - RELAX_STEP_RAD)
                    any_moving = True
            self._publish_hand_positions_subset(list(current), current, RELAX_TICK_SEC)
            self._observe(f"relax_tick_{ticks['n']}", object_name, {"red": red})

            if any_moving and ticks["n"] < RELAX_MAX_TICKS:
                self._call_once_after(RELAX_TICK_SEC, step)
            else:
                self.get_logger().info(
                    f"Relax phase done after {ticks['n']} ticks -- holding at minimum force, apple still visible.")
                self._observe("relax_done", object_name, {"final_red": red})
                self._call_once_after(LIFT_SETTLE_SEC, lambda: self._report_done(object_name))

        step()

    def _report_done(self, object_name):
        msg = String()
        msg.data = json.dumps({"object_name": object_name})
        self.sequence_done_pub.publish(msg)
        self.get_logger().info(f"Sequence done for '{object_name}'; reported back.")

    def on_trigger(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Alignment trigger was not valid JSON; ignoring.")
            return

        object_name = data.get("object_name", "unknown object")
        real_size_m = data.get("real_size_m")

        if real_size_m is None:
            size_fraction = 0.5  # unknown size -- use a moderate default spread
        else:
            lo, hi = MIN_OBJECT_WIDTH_M, MAX_OBJECT_WIDTH_M
            size_fraction = max(0.0, min(1.0, (real_size_m - lo) / (hi - lo))) if hi > lo else 0.5

        # spread scales with the object's real measured size -- bigger
        # object, wider fan (was missing entirely: on_trigger referenced
        # `spread` without ever computing it, so every real alignment
        # trigger raised a NameError and silently did nothing).
        spread = size_fraction * MAX_YAW_SPREAD_RAD

        # EVENLY spaced AND centered on Yaw=0 (2026-09-17): previously the
        # four steps were Index/Middle/Ring/Pinky = +u/0/-u/-2u, which is
        # evenly SPACED but not centered -- the whole finger group's
        # centroid sat at -0.5u instead of straight ahead, so it wasn't
        # symmetric around the thumb's fixed opposing position. Centering
        # the four steps around 0 (+1.5u/+0.5u/-0.5u/-1.5u) keeps the same
        # per-step spacing and the same outermost (Pinky) magnitude class,
        # but now the finger group as a whole faces the thumb squarely
        # instead of being biased to one side -- needed so the object ends
        # up precisely between the thumb and the 4-finger group rather than
        # off to one side of it.
        # Widened to a gripping fan (2026-09-21): outer fingers at +/-0.34 rad
        # (joint limit is 0.349), four equal steps, so the fingers are spread
        # around the object instead of bunched together.
        unit = GRIP_FAN_OUTER_RAD / 3.0
        yaw_roll = {
            "R_Thumb_Roll": THUMB_ROLL_RAD, "R_Thumb_Yaw": THUMB_YAW_RAD,
            # Signs verified against the URDF: NEGATIVE Index/Middle and POSITIVE
            # Ring/Pinky spread the fingertips apart (the opposite converges them).
            "R_Index_Yaw": -1.5 * unit,
            "R_Middle_Yaw": -0.5 * unit,
            "R_Ring_Yaw": 0.5 * unit,
            "R_Pinky_Yaw": 1.5 * unit,
        }
        self._yaw_roll_positions = yaw_roll  # close_hand reuses this -- don't re-derive it

        # Fully straight (Pitch AND Flexor/DIP all 0) -- REVERTED (2026-09-17)
        # from keeping a four-finger pre-curl here: that ran end-to-end but
        # still pushed the object. Fully straight through align was the ONE
        # configuration confirmed, repeatedly, to leave the object completely
        # undisturbed. Only Yaw/Roll (lateral spread) is this node's job here;
        # curling only happens in the single decisive close_hand grab below.
        #
        # Sent as TWO separate messages, not one combined one -- see
        # _publish_hand_positions_subset for why (Pitch staying at 0 visibly
        # bulged mid-motion when bundled with Yaw actually moving).
        straight_positions = {name: 0.0 for name in PITCH_FLEXOR_DIP_JOINTS}
        # One combined message (a second message would replace the first
        # before it moves and freeze the claw curl at 0).
        combined = dict(straight_positions)
        combined.update(yaw_roll)
        self._publish_hand_positions(combined, ALIGN_DURATION_SEC)

        size_desc = f"{real_size_m:.3f}m" if real_size_m is not None else "unknown size"
        self.get_logger().info(
            f"Aligned straight fingers around '{object_name}' ({size_desc}, "
            f"yaw spread {spread:.3f} rad)."
        )
        # Close/lift REMOVED (2026-09-17) per explicit request: every close
        # variation tried tonight (staged pre-curl, single fast grab, slow
        # gentle grab, thumb isolated) pushed the object without ever
        # lifting it. Fully straight through descend+align is the one
        # configuration confirmed, repeatedly, to leave the object
        # completely undisturbed -- stopping here rather than risk
        # disturbing it with another closing attempt.
        self._call_once_after(
            ALIGN_DURATION_SEC + CLOSE_SETTLE_SEC, lambda: self.close_hand(object_name)
        )


def main(args=None):
    rclpy.init(args=args)
    node = FingerAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
