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
from ikpy.chain import Chain
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanded_robot.urdf")
# Same hand-frame convention as vision_ik_overhead_closed_loop.py (duplicated,
# these two processes don't share state): ikpy's end-effector frame is offset
# 180deg about X from the hand's own convenient axes.
HAND_FRAME_FROM_IK_FRAME = np.diag([1.0, -1.0, -1.0])
APPLE_RADIUS_M = 0.04  # must match vision_ik_overhead_closed_loop.py's APPLE_HALF_HEIGHT_M

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
OBSERVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "grasp_observations")

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
LIFT_SHOULDER_LIFT_DELTA_RAD = -0.07  # (unused now -- see LIFT_HEIGHT_M) single-joint delta only moved the wrist ~3cm, nowhere near the user-required 15cm; replaced by an IK-solved pure-vertical translation (2026-10-14)
LIFT_DURATION_SEC = 14.0  # slower lift to match the gentler delta
LIFT_HEIGHT_M = 0.15  # user requirement: lift 15cm
LIFT_HOLD_CHECK_SEC = 3.0  # how long the apple must stay with the hand to pass
LIFT_HOLD_TICK_SEC = 0.5
LIFT_XY_TOLERANCE_M = 0.02  # user requirement: apple stays within 2cm (horizontally) of the grasp point

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
CLOSE_AMOUNT_PITCH_FLEXOR = 0.9  # pushed further -- thumb now reaches cleanly and gently, fingers still fall short
CLOSE_AMOUNT_DIP = 0.65
# DIAGNOSTIC (2026-09-17): every attempt tonight pushed the object in the
# same direction (increasingly negative Y) regardless of amount or speed --
# a systematic bias, not random contact noise. The thumb travels the
# largest arc of any joint (starts fully extended, has the most range to
# sweep through) and is the prime suspect. Isolating it: thumb stays fully
# extended through the whole close, only the four fingers move, to see
# whether the systematic push goes away.
CLOSE_AMOUNT_THUMB_PITCH_FLEXOR = 0.8
CLOSE_AMOUNT_THUMB_DIP = 0.55

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
THUMB_YAW_RAD = -0.35  # opposed/close-ready value -- only reached gradually during close_hand now
THUMB_ROLL_RAD = -0.20  # rotated further toward opposing the fingers (was -0.10)
# Must match vision_ik_overhead_closed_loop.py's PRE_DESCENT_THUMB_YAW_RAD /
# PRE_DESCENT_THUMB_ROLL_RAD -- the thumb's pose is set there once, before
# descent, and this node must know what it already is to animate FROM it
# during close (duplicated since the two processes don't share state).
PRE_DESCENT_THUMB_YAW_RAD = -0.5
PRE_DESCENT_THUMB_ROLL_RAD = -0.34
PRE_DESCENT_THUMB_CURL = 0.65  # must match vision_ik_overhead_closed_loop.py's constant of the same name


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
        self._pre_lift_apple = None
        self._lift_hold_ok_streak = 0

        # ikpy chain (2026-10-14): needed for an IK-solved pure-vertical lift
        # (see lift()) and for checking real fingertip/thumb-tip positions
        # against the apple's ground truth after closing (see
        # _closed_fingertips_world). Duplicated from
        # vision_ik_overhead_closed_loop.py since these two processes don't
        # share state.
        self.chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base_link"])
        self.active_mask = [False] * len(self.chain.links)
        for i in range(2, 8):
            if i < len(self.active_mask):
                self.active_mask[i] = True
        self.chain.active_links_mask = self.active_mask

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

    def _apple_ground_truth(self):
        """Real apple position from the sim (not vision), via the same `ign
        topic` method established earlier this project for ground-truth
        verification. Returns [x,y,z] in world frame, or None on failure."""
        try:
            out = subprocess.run(
                "ign topic -e -t /world/apple_world/pose/info -n 1",
                shell=True, capture_output=True, text=True, timeout=10,
            ).stdout
            m = re.search(r'name: "apple_05".*?position \{(.*?)\}', out, re.S)
            vals = dict(re.findall(r'([xyz]): ([-\d.e]+)', m.group(1))) if m else {}
            return [round(float(vals.get(k, 0.0)), 4) for k in "xyz"]
        except Exception:
            return None

    def _arm_fk(self):
        """FK of the arm's current /joint_states, in the same frame ikpy
        uses. Returns (wrist_pos, wrist_rot_ik_frame, joint_array) or None if
        joint states aren't available yet."""
        for name in ARM_JOINTS:
            if name not in self.latest_arm_positions:
                return None
        ja = [0.0] * len(self.chain.links)
        for i, name in enumerate(ARM_JOINTS):
            ja[2 + i] = self.latest_arm_positions[name]
        fk = self.chain.forward_kinematics(ja)
        return fk[:3, 3], fk[:3, :3], ja

    def _closed_fingertips_world(self, hand_positions):
        """FK for each finger's tip and the thumb tip given ACTUAL commanded
        joint values (not assumed close-ready ones), transformed into world
        frame via the arm's current FK. Used to verify fingertips end up
        BELOW the apple's centroid once closed (point 3, user 2026-10-14),
        and to estimate the open-hand span (point 4). Joints missing from
        hand_positions default to 0.0 (straight/open)."""
        import xml.etree.ElementTree as ET

        def rpy_mat(r):
            a, b, c = r
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        def axis_mat(ax, th):
            x, y, z = ax / np.linalg.norm(ax)
            c, s = np.cos(th), np.sin(th)
            C = 1 - c
            return np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                              [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
                              [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])

        root = ET.parse(URDF_PATH).getroot()
        J = {}
        for j in root.findall("joint"):
            n = j.get("name")
            if not n or not n.startswith("R_"):
                continue
            o = j.find("origin")
            J[n] = (np.array([float(x) for x in o.get("xyz").split()]),
                    np.array([float(x) for x in o.get("rpy").split()]),
                    np.array([float(x) for x in j.find("axis").get("xyz").split()]))

        def tip(chain, jv):
            T = np.eye(4)
            for n in chain:
                xyz, r, ax = J[n]
                M = np.eye(4)
                M[:3, :3] = rpy_mat(r) @ axis_mat(ax, jv.get(n, 0.0))
                M[:3, 3] = xyz
                T = T @ M
            return T[:3, 3]

        tips_local = {}
        for name in ("Index", "Middle", "Ring", "Pinky"):
            chain = [f"R_{name}_Yaw", f"R_{name}_Pitch", f"R_{name}_Flexor", f"R_{name}_DIP"]
            tips_local[name] = tip(chain, hand_positions)
        thumb_chain = ["R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch", "R_Thumb_Flexor", "R_Thumb_DIP"]
        tips_local["Thumb"] = tip(thumb_chain, hand_positions)

        fk = self._arm_fk()
        if fk is None:
            return None
        wrist_pos, wrist_rot, _ = fk
        r_hand = wrist_rot @ HAND_FRAME_FROM_IK_FRAME
        return {name: wrist_pos + r_hand @ local for name, local in tips_local.items()}

    def _observe(self, phase, object_name, extra=None):
        """Save a gripper_camera + side_camera frame and log joint efforts /
        red-fraction, tagged with the phase name, so a run can be reviewed
        frame-by-frame afterward instead of only from the final result."""
        stamp = time.strftime("%H%M%S")
        red = self._red_fraction()
        efforts = {n: round(self.latest_hand_effort.get(n, 0.0), 2) for n in PITCH_FLEXOR_DIP_JOINTS}
        apple_ground_truth = self._apple_ground_truth()
        if apple_ground_truth is None:
            apple_ground_truth = "ERROR: no reading"
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
            current[name] = PRE_DESCENT_THUMB_CURL if name.startswith("R_Thumb") else 0.0
            frozen[name] = False
            hit_count[name] = 0
            squeeze_ticks[name] = 0
        ticks = {"n": 0}

        # Thumb Yaw/Roll swing happens HERE now, as part of the close motion
        # (user: "the only finger command after descent is the close") --
        # starts from the pre-descent abducted pose, ramps toward opposed
        # over the same tick cadence as the curl, included in every
        # published message below so it's one continuous close, not a
        # separate pre-close snap.
        thumb_yr_current = {"R_Thumb_Yaw": PRE_DESCENT_THUMB_YAW_RAD, "R_Thumb_Roll": PRE_DESCENT_THUMB_ROLL_RAD}
        thumb_yr_target = {"R_Thumb_Yaw": THUMB_YAW_RAD, "R_Thumb_Roll": THUMB_ROLL_RAD}

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

            for name in thumb_yr_current:
                if thumb_yr_current[name] < thumb_yr_target[name]:
                    thumb_yr_current[name] = min(thumb_yr_target[name], thumb_yr_current[name] + CLOSE_STEP_RAD)
                    any_moving = True
                elif thumb_yr_current[name] > thumb_yr_target[name]:
                    thumb_yr_current[name] = max(thumb_yr_target[name], thumb_yr_current[name] - CLOSE_STEP_RAD)
                    any_moving = True

            publish_positions = dict(current)
            publish_positions.update(thumb_yr_current)
            self._publish_hand_positions_subset(list(publish_positions), publish_positions, CLOSE_TICK_SEC)
            if any_moving and ticks["n"] < CLOSE_MAX_TICKS:
                self._call_once_after(CLOSE_TICK_SEC, step)
            else:
                # Count DISTINCT fingers touched, not joints -- a single
                # finger has 3 joints (Pitch/Flexor/DIP) that can each
                # register contact, which previously miscounted "1 finger
                # touching" as "3 finger(s) in contact" and let a grasp with
                # only the pinky actually on the apple proceed to lift.
                # Confirmed live 2026-10-14: exactly this happened (only
                # R_Pinky_* touched, logged as "3 finger(s)", apple never
                # left the table on lift).
                contact_links = [n for n in squeeze_ticks if squeeze_ticks[n] > 0]
                touched_finger_names = {n.split("_")[1] for n in contact_links if not n.startswith("R_Thumb")}
                fingers_touched = len(touched_finger_names)
                thumb_touched = any(n.startswith("R_Thumb") for n in contact_links)
                self._secure_positions = dict(current)  # relax phase starts from here
                self.get_logger().info(f"[verify] contact links: {contact_links}")
                if fingers_touched >= 2 and thumb_touched:
                    self.get_logger().info(
                        f"Grab converged on '{object_name}' after {ticks['n']} ticks -- "
                        f"thumb AND {fingers_touched} distinct finger(s) in contact ({sorted(touched_finger_names)}) "
                        f"(SECURE phase); lifting.")
                    closed_positions = dict(current)
                    closed_positions.update(thumb_yr_current)
                    tips_world = self._closed_fingertips_world(closed_positions)
                    apple_gt = self._apple_ground_truth()
                    if tips_world and apple_gt:
                        apple_z = apple_gt[2]
                        for name, pos in tips_world.items():
                            below = pos[2] < apple_z
                            self.get_logger().info(
                                f"[verify] {name} tip z = {pos[2]:.3f}m vs apple centroid z = {apple_z:.3f}m: "
                                f"{'BELOW (OK)' if below else 'ABOVE -- will slip out on lift'}"
                            )
                    self._observe("secure_grip_done", object_name, {"contact_links": contact_links})
                    self._call_once_after(LIFT_SETTLE_SEC, lambda: self.lift(object_name))
                else:
                    self.get_logger().warning(
                        f"NOT lifting '{object_name}' -- apple not confirmed in hand "
                        f"(thumb_touched={thumb_touched}, fingers_touched={fingers_touched} {sorted(touched_finger_names)}, "
                        f"need >=2 distinct fingers AND thumb).")
                    self._observe("secure_grip_failed", object_name,
                                  {"thumb_touched": thumb_touched, "fingers_touched": fingers_touched,
                                   "contact_links": contact_links})
                    self._call_once_after(LIFT_SETTLE_SEC, lambda: self._report_done(object_name))

        self._observe("pre_close", object_name)
        step()

    def lift(self, object_name):
        """Raise the arm LIFT_HEIGHT_M (15cm, user requirement) to test
        whether the grasp actually holds. REPLACED (2026-10-14) the old
        single-joint shoulder_lift delta -- that only moved the wrist ~3cm
        for its -0.07 rad nudge, nowhere near 15cm, and a bigger single-joint
        delta would also drag the wrist sideways, not purely up. Now solves
        a real IK target (current wrist position + 0.15m in world Z, same
        orientation), warm-started from the current joint angles so it stays
        on the same configuration branch instead of jumping."""
        fk = self._arm_fk()
        if fk is None:
            self.get_logger().warning("No joint_states yet for arm joints; skipping lift.")
            return
        wrist_pos, wrist_rot, seed_ja = fk
        target_pos = wrist_pos + np.array([0.0, 0.0, LIFT_HEIGHT_M])
        try:
            sol = self.chain.inverse_kinematics(
                target_position=target_pos, target_orientation=wrist_rot,
                orientation_mode="all", initial_position=seed_ja,
            )
        except Exception as e:
            self.get_logger().error(f"Lift IK solve failed: {e}; skipping lift.")
            return
        positions = [sol[2 + i] for i in range(6)]

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(LIFT_DURATION_SEC)
        point.time_from_start.nanosec = int((LIFT_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.arm_traj_pub.publish(traj)
        self.get_logger().info(f"Lifting {LIFT_HEIGHT_M*100:.0f}cm (IK-solved vertical) to test the grasp on '{object_name}'.")
        self._pre_lift_apple = self._apple_ground_truth()
        self._lift_hold_ok_streak = 0
        self._observe("lift_commanded", object_name)
        self._call_once_after(LIFT_DURATION_SEC + LIFT_SETTLE_SEC, lambda: self._check_lift_held(object_name, 0))

    def _check_lift_held(self, object_name, tick):
        """User requirement: pass only if the apple rises with the hand and
        stays within 2cm (horizontally) of its pre-lift position for 3s,
        checked against GROUND TRUTH (ign topic), not the vision-only
        red_fraction signal the old check relied on."""
        apple = self._apple_ground_truth()
        pre = self._pre_lift_apple
        if apple is None or pre is None:
            self.get_logger().warning(f"[lift check] tick {tick}: no ground-truth reading -- treating as FAIL.")
            ok = False
            xy_drift = dz = None
        else:
            dx, dy, dz = apple[0] - pre[0], apple[1] - pre[1], apple[2] - pre[2]
            xy_drift = (dx ** 2 + dy ** 2) ** 0.5
            ok = xy_drift <= LIFT_XY_TOLERANCE_M and dz >= LIFT_HEIGHT_M * 0.5
            self.get_logger().info(
                f"[lift check] tick {tick}: apple={apple} xy_drift={xy_drift*1000:.1f}mm "
                f"(need <={LIFT_XY_TOLERANCE_M*1000:.0f}mm) dz={dz*1000:.1f}mm "
                f"(need >={LIFT_HEIGHT_M*500:.0f}mm): {'OK' if ok else 'FAIL'}"
            )
        self._observe(f"lift_check_{tick}", object_name, {"apple_ground_truth": apple, "xy_drift_m": xy_drift, "dz_m": dz, "ok": ok})
        if not ok:
            self._lift_hold_ok_streak = 0
            self.get_logger().warning(
                f"LIFT CHECK FAILED at tick {tick} -- apple did not stay with the hand "
                f"within {LIFT_XY_TOLERANCE_M*100:.0f}cm while rising >= {LIFT_HEIGHT_M*50:.0f}cm."
            )
            self._call_once_after(LIFT_SETTLE_SEC, lambda: self.relax_grip(object_name))
            return
        self._lift_hold_ok_streak += 1
        if self._lift_hold_ok_streak * LIFT_HOLD_TICK_SEC >= LIFT_HOLD_CHECK_SEC:
            self.get_logger().info(
                f"LIFT CHECK PASSED: apple stayed within {LIFT_XY_TOLERANCE_M*100:.0f}cm of its pre-lift "
                f"position for {LIFT_HOLD_CHECK_SEC:.1f}s while rising with the hand."
            )
            self._call_once_after(LIFT_SETTLE_SEC, lambda: self.relax_grip(object_name))
            return
        self._call_once_after(LIFT_HOLD_TICK_SEC, lambda: self._check_lift_held(object_name, tick + 1))

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
        # THUMB EXCLUDED from this message (2026-10-14, user: "no finger or
        # thumb opening command may be sent once the descent starts or after
        # it ends -- the only finger command after descent is the close").
        # Confirmed live this align-time swing (from the descent-safe
        # abducted pose to this opposed one) was itself what hit the apple
        # -- the very first close tick showed effort 15.6 on the thumb
        # before it had moved at all, meaning it was already touching right
        # after this swing. The thumb now stays at its pre-descent pose
        # (set once, before descent, held through descent AND align) and
        # only moves as part of close_hand's close motion below.
        yaw_roll = {
            # Signs verified against the URDF: NEGATIVE Index/Middle and POSITIVE
            # Ring/Pinky spread the fingertips apart (the opposite converges them).
            "R_Index_Yaw": -1.5 * unit,
            "R_Middle_Yaw": -0.5 * unit,
            "R_Ring_Yaw": 0.5 * unit,
            "R_Pinky_Yaw": 1.5 * unit,
        }
        self._yaw_roll_positions = dict(yaw_roll)
        self._yaw_roll_positions["R_Thumb_Roll"] = PRE_DESCENT_THUMB_ROLL_RAD
        self._yaw_roll_positions["R_Thumb_Yaw"] = PRE_DESCENT_THUMB_YAW_RAD

        # Fingers only (thumb Pitch/Flexor/DIP also excluded -- it keeps its
        # pre-descent curl here, straightens only during the close below).
        straight_positions = {name: 0.0 for name in PITCH_FLEXOR_DIP_JOINTS if not name.startswith("R_Thumb")}
        combined = dict(straight_positions)
        combined.update(yaw_roll)
        self._publish_hand_positions_subset(list(combined.keys()), combined, ALIGN_DURATION_SEC)

        size_desc = f"{real_size_m:.3f}m" if real_size_m is not None else "unknown size"
        self.get_logger().info(
            f"Aligned straight fingers around '{object_name}' ({size_desc}, "
            f"yaw spread {spread:.3f} rad)."
        )

        # User requirement (2026-10-14, point 4): report the open-hand span
        # vs the apple's diameter. This is measured at the fully-open/straight
        # pose, so it's a conservative lower bound -- actual contact happens
        # partway through the close sweep (see GRASP_POINT_IN_HAND_M's
        # derivation notes in vision_ik_overhead_closed_loop.py), so a span
        # smaller than the apple's diameter here does NOT by itself mean the
        # grasp is infeasible.
        open_positions = {"R_Thumb_Yaw": PRE_DESCENT_THUMB_YAW_RAD, "R_Thumb_Roll": PRE_DESCENT_THUMB_ROLL_RAD}
        open_positions.update(yaw_roll)
        open_tips = self._closed_fingertips_world(open_positions)
        if open_tips:
            span = float(np.linalg.norm(open_tips["Thumb"] - open_tips["Middle"]))
            apple_diam = 2 * APPLE_RADIUS_M
            self.get_logger().info(
                f"[verify] open-hand span (thumb to middle fingertip) = {span:.3f}m, "
                f"apple diameter = {apple_diam:.3f}m."
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
