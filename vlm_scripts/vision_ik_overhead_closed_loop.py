#!/usr/bin/env python3
import json
import math
import os
import subprocess
import time
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from ikpy.chain import Chain

FRAGILITY_TOPIC = "/overhead_camera/fragility_analysis"
ALIGNMENT_REQUEST_TOPIC = "/overhead_camera/alignment_check_request"
ALIGNMENT_RESULT_TOPIC = "/overhead_camera/alignment_check_result"
ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"
# adaptive_grasp_controller.py listens here and takes over actual finger
# closing with real-time joint-effort feedback (squeeze until deformation
# onset, back off, lock) the moment it sees a new object_name -- this node
# hands off to it instead of closing the hand itself with one fixed-force
# trajectory. Both nodes publish to /dexhand_controller/joint_trajectory, but
# never at the same time: this node only ever sends an OPEN-hand pre-shape
# (see open_hand_preshape) before handoff, never a close, so there's no
# fight over the topic once adaptive_grasp_controller.py takes over.
GRIPPER_FRAGILITY_TOPIC = "/gripper_camera/fragility_analysis"
# finger_alignment_node.py listens here: once the arm is CONFIRMED (via FK
# feedback, not a fixed timer) to have actually reached the target, this
# node's job ends -- it hands off object_name/position/size and lets that
# separate node do straight-finger (no curl) positioning around the object
# using its own vision-based judgement. No squeeze/curl/adaptive-controller
# handoff happens in THIS file anymore.
FINGER_ALIGN_TRIGGER_TOPIC = "/pick/ready_for_finger_alignment"
FINGER_SEQUENCE_DONE_TOPIC = "/pick/finger_sequence_done"
# Generous: under this sim's observed real-time-factor (~0.1x, confirmed live
# 2026-09-16 -- a nominal 3s wait actually took ~31s), align+close+lift can
# take a couple of minutes of wall-clock time. This is only a safety net for
# if finger_alignment_node.py's own report never arrives.
FINGER_SEQUENCE_TIMEOUT_SEC = 180.0
HAND_TRAJECTORY_TOPIC = "/dexhand_controller/joint_trajectory"
# FAST on purpose (2026-09-17): this fires at the same moment as the arm's
# own MOVE_DURATION_SEC=4.0s descent. Under this sim's real-time-factor
# (visibly ~0.1x all session), a "slow" 1.5s hand-open trajectory actually
# takes ~15 real seconds to finish -- so for a big chunk of the descent the
# fingers are still transitioning from whatever curl they ended the PREVIOUS
# cycle's close at, not actually straight yet. Confirmed live: this is what
# looked like "descending pre-curled" even though the commanded target was
# straight the whole time. Snapping the hand open fast means it's genuinely
# straight well before the arm gets moving, not racing it.
HAND_OPEN_DURATION_SEC = 0.3
# Four fingers hang in a curved claw (knuckle/middle/tip, raw rad); thumb stays straight.
CLAW_RAD = {"Pitch": 0.5, "Flexor": 0.8, "DIP": 0.5}
# Thumb Roll/Yaw here use PRE_DESCENT_THUMB_* (swept out to the side, least
# protrusion below the palm) -- the opposed values for closing are applied
# separately by finger_alignment_node.py's align step, once the palm is
# already low (see APPROACH_HEIGHT_ABOVE_M / the 2-stage descent above).
PRE_DESCENT_FAN_RAD = {
    "R_Index_Yaw": -0.14, "R_Middle_Yaw": -0.047, "R_Ring_Yaw": 0.047, "R_Pinky_Yaw": 0.14,
}

HAND_JOINTS = [
    "R_Thumb_Pitch", "R_Thumb_Roll", "R_Thumb_Yaw", "R_Thumb_Flexor", "R_Thumb_DIP",
    "R_Index_Pitch", "R_Index_Yaw", "R_Index_Flexor", "R_Index_DIP",
    "R_Middle_Pitch", "R_Middle_Yaw", "R_Middle_Flexor", "R_Middle_DIP",
    "R_Ring_Pitch", "R_Ring_Yaw", "R_Ring_Flexor", "R_Ring_DIP",
    "R_Pinky_Pitch", "R_Pinky_Yaw", "R_Pinky_Flexor", "R_Pinky_DIP",
]
CLOSING_JOINTS = {j for j in HAND_JOINTS if j.endswith("Flexor") or j.endswith("DIP")}
# Pitch also curls a finger toward the palm under this hand's convention
# (not just Flexor/DIP) -- confirmed live 2026-09-16: pre-curling Pitch
# during the arm's descent was catching/pushing the object out of the way
# before the hand even finished settling. Treated the same as Flexor/DIP:
# stays fully open through open_hand_preshape, only curls in afterward via
# align_fingers_to_object once the hand has actually reached the target.
PITCH_JOINTS = {j for j in HAND_JOINTS if j.endswith("Pitch")}
FINGER_ALIGN_DURATION_SEC = 1.5
FINGER_ALIGN_SETTLE_SEC = 0.5

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanded_robot.urdf")

WORLD_FRAME = "base_footprint"
BASE_FRAME = "base_link"

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

GRASP_COOLDOWN_SEC = 90.0
GRASP_PROXIMITY_PX = 40.0

CAMERA_WORLD_X = 1.125
CAMERA_WORLD_Y = 0.0
CAMERA_WORLD_Z = 2.0

IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
HORIZONTAL_FOV_RAD = 1.396
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)

APPLE_WORLD_Z = 0.05

# SIGN CONFIRMED (2026-09-16) via live ground truth in yolo_depth_detector.py:
# apple_05 spawned at world (1.0, 0.0, 0.04) actually appeared at overhead
# pixel (320, 266), not the (320, 216) that U_SIGN=V_SIGN=+1 predicts -- only
# -1 reproduces the real position. The old comment claiming [0,0,1] was
# "confirmed via manual_target_node.py testing" was about the approach
# *orientation*, not this pixel-to-world mapping; this mapping was never
# independently validated until now. Both axes flip together (same 90-degree
# camera-optical-axis rotation).
U_SIGN = -1.0
V_SIGN = -1.0

# SIDE / HANDSHAKE GRASP (2026-10-09, user request): replaces the top-down
# approach. Top-down meant the fingertips reached the ground before the palm
# got anywhere near the apple (confirmed by direct screenshot), so the apple
# ended up beside the fingers, never inside the grasp.
#
# Derived from real geometry, not guessed:
#  - Finger Pitch joints rotate about local Y (URDF axis vectors ~(1,0,0) for
#    all 4 non-thumb fingers) -> local Y is the thumb-to-pinky lateral axis.
#  - Palm mesh bounding box (base_link.stl, scale 0.001): Y spans -0.0539
#    (pinky-side edge) to +0.0455 (thumb/index side).
#  - Local Z is the finger-reach axis (straight fingertips sit at Z~0.16).
# Handshake pose: local Y -> world Z (pinky/-Y edge down, thumb/+Y side up),
# local X (palm normal) -> horizontal toward the apple (+X_base), local Z
# (fingers) -> the remaining horizontal axis. ex=(1,0,0), ey=(0,0,1),
# ez=ex x ey=(0,-1,0) in base_link frame.
TARGET_APPROACH_DIRECTION = [0, 0, 1]  # unused now (orientation_mode="all" below) -- kept for the old "Z" fallback path
# "all" + a full rotation matrix (tried 2026-09-16, using pocket_grasp_
# test.py's own measured palm-down + 45deg tilt orientation, ported from the
# user's ur_gz_apple_gripper repo) worked for the first target it was tried
# on but then produced a badly wrong orientation on a later one. Root cause:
# fully constraining both position (3) AND orientation (3) on a 6-joint arm
# leaves the numerical solver zero spare freedom, so it needs a GOOD seed
# for every target to reliably converge -- pocket_grasp_test.py's own
# solve_ik tries ~13 different seed presets and keeps whichever converges;
# this file only tried one. Back to "Z" (free roll, reliable all night)
# rather than ship an unreliable fully-constrained solve.
ORIENTATION_MODE = "all"
# R_ik = R_hand @ HAND_FRAME_FROM_IK_FRAME (that matrix is self-inverse, diag
# +-1), computed from the R_hand above. See the comment block for the axis
# mapping. Verified live (2026-10-09): matches the actual achieved FK
# orientation to within the solver's own residual.
# TOP-DOWN, PALM-DOWN (2026-10-09, user request -- replaces the side grasp
# above): local X (palm normal) -> world -Z (straight down); local Z
# (finger/reach axis) -> horizontal +X_base ("forward"); local Y (lateral,
# thumb-to-pinky) -> the remaining horizontal axis. Derived the same way as
# the side grasp: from the finger Pitch-joint rotation axis (local Y) and
# the palm mesh's own bounding box, not guessed.
# BUG FOUND AND FIXED (2026-10-09): this held R_hand (the real hand
# rotation) instead of R_ik = R_hand @ HAND_FRAME_FROM_IK_FRAME, which is
# what ikpy's target_orientation actually needs (the chain's own last-link
# frame, ft_frame, sits 180deg-about-X rotated from the real hand -- see the
# live TF check: wrist_3_link->ft_frame quat=(1,0,0,0) vs
# wrist_3_link->dexhand_base_link quat=(0,0,0,1)). Passing R_hand directly
# sent the solver chasing a target it could never actually reach, so the
# hand-frame correction loop diverged instead of converging (155mm
# residual, confirmed live and reproduced offline).
SIDE_GRASP_TARGET_ORIENTATION = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
])
# Alternate arm seeds to try for the full-orientation solve, in order, keeping
# whichever converges with the lowest position+orientation residual -- a
# single seed was flagged earlier in this project as unreliable for fully
# constrained (position+orientation) IK on this 6-DOF arm; this reach is also
# a very different arm configuration (sideways, not top-down) from the seed
# that was tuned for the old approach, so the risk is real here.
IK_ALT_SEEDS = [
    [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
    [1.2, -1.2, 1.5, -1.8, -1.57, 0.0],
    [1.2, -0.8, 1.8, -2.5, -1.2, 0.0],   # ELBOW-UP: confirmed live (2026-10-09) this one finds elbow~0.5m / wrist_1~0.24m above ground, vs ~0.235m/0.04m for the others -- the others were letting the forearm nearly touch the ground.
    [0.6, -1.4, 1.9, -2.1, -1.9, 0.6],
    [1.57, -1.0, 1.2, -1.7, -1.57, -0.5],
    [1.0, -0.6, 1.9, -2.3, -1.3, 0.2],   # ELBOW-UP backup, in case the one above doesn't converge for a given target
]

# User fix request (2026-10-09): "elbow and forearm touch the ground, blocks
# the hand from lowering" -- confirmed live, most seeds above converge to
# wrist_1_joint only 4cm above the floor. Any IK candidate with elbow_joint
# or wrist_1_joint (the forearm's two ends) below this height is rejected
# outright during seed selection, not just deprioritized -- see solve_ik.
SAFE_ARM_HEIGHT_ABOVE_GROUND_M = 0.10
ARM_SAFETY_CHECK_LINKS = ("elbow_joint", "wrist_1_joint")
# base_footprint (ground, world Z=0) -> mobile_base_link: z=0 (fixed joint).
# mobile_base_link -> base_link: z=+0.285 (fixed joint). Confirmed directly
# from the URDF, not assumed -- base_link sits 0.285m above the ground.
BASE_LINK_HEIGHT_ABOVE_GROUND_M = 0.285

# REVERTED (2026-09-17): tried porting pocket_grasp_test.py's tilted-approach-
# specific pocket-aim offset (AIM_DEPTH/PALM_OFFSET/LATERAL/GRASP_DROP, their
# own measured values) plus a matching 0.25 rad pre-curl. Ran end-to-end
# (real contact detected, all stages completed) but still pushed the object
# ~0.8m with no lift -- those offset numbers were measured for THEIR specific
# pre-curled, 45deg-tilted approach geometry, which this file no longer uses
# (ORIENTATION_MODE reverted to "Z", no tilt), so mixing their tuned numbers
# with a different overall strategy was never consistent. Back to the
# simpler, static HAND_OFFSET_VECTOR_M -- less sophisticated (doesn't
# rotate with whatever roll ikpy picks) but it's what produced the ONLY
# repeatedly-confirmed zero-disturbance result all night, with fully
# straight (no pre-curl) fingers through the whole descend+align.
# Z reduced (2026-09-17) per visual feedback: the gripper was hovering with
# a visible gap above the object. Smaller Z offset here -> wrist_target
# (target_base - this vector) sits lower/closer, bringing the hand nearer
# the object without touching the X/Y alignment.
# REVERTED (2026-09-17): the FK-derived retune above (and the Z-only
# correction after it) both made the visible gap WORSE, confirmed by two
# separate live screenshots -- IK doesn't move each offset axis
# independently; a small change to one component of this vector shifts the
# whole arm's solved configuration (including its rotation, since
# ORIENTATION_MODE="Z" leaves azimuth free), so the derived-from-FK math
# didn't transfer cleanly into an actual improvement. Going back to the
# value that was live-confirmed working earlier tonight ("bring closer")
# rather than continuing to guess from an under-modeled correction.
HAND_OFFSET_VECTOR_M = (0.0, 0.0, 0.0)  # only the first IK guess -- iteration in solve_ik_hand_frame corrects it, doesn't need to be precise
# Where the object centre should sit in the hand's own frame (dexhand_base_link):
# X spans thumb (~0.10) to fingers (~0.006), Z runs along the fingers.
# SIDE GRASP grasp point (2026-10-09): local X,Z (palm-normal and finger-
# reach) are a starting estimate -- same order of magnitude as the digits'
# own reach measured earlier this project (thumb ~0.07-0.10m, fingers up to
# ~0.16m) -- to be refined by the same measure-and-correct method used all
# along. Local Y is NOT a guess: computed from the palm mesh's real bounding
# box and the apple's known resting height so the apple lands at its own
# true world height while the palm's pinky-edge grazes the ground (see the
# ORIENTATION_MODE="all" comment block above for the full derivation).
# Palm-down grasp point (2026-10-09): places the PALM LINK'S OWN GEOMETRIC
# CENTER (from the STL bounding box average, not dexhand_base_link's origin
# -- the user specifically asked for the palm link, not tool0/wrist) at a
# standoff of apple-half-height + 0.75cm directly above the apple, with zero
# horizontal offset (Y,Z components are the palm center's own local Y,Z --
# horizontal alignment needs no correction once the rotation is right).
GRASP_POINT_IN_HAND_M = (0.081, 0.002, 0.110)  # recomputed 2026-10-14 via FK sweep of the REAL close-motion trajectory (finger_alignment_node.py's actual CLOSE_AMOUNT_* targets, scaled by each joint's real URDF limits -- not raw radians). The old point's Z (0.0407, the palm's own STL bounding-box Z) put the apple 7cm short of where any finger or thumb ever reaches during closing -- confirmed live: only the pinky ever touched, apple never lifted. New point is where all 4 fingertips (close_frac 0.54-0.64) and the thumb tip (close_frac 0.15-0.27, either side of its yaw/roll swing) are verified to pass within the apple's radius (0.04m) during the real close sweep -- see /tmp scratchpad hand_geometry_sweep4/5.py for the derivation.
APPLE_HALF_HEIGHT_M = 0.04  # current apple collision box is a 0.08m cube
APPROACH_HEIGHT_ABOVE_M = 0.10  # waypoint 1: this far above the final pre-close pose, same horizontal position and orientation
# user request (2026-10-09): raised to the 10-15cm range explicitly, and the
# descent is now a genuinely re-solved multi-waypoint Cartesian-straight
# path (each sample independently IK-solved for the SAME fresh x,y), not a
# hope that joint-space interpolation between 2 points stays vertical.
APPROACH_HEIGHT_ABOVE_M = 0.13
DESCENT_WAYPOINTS = 5  # intermediate IK solves from approach height down to the final pre-close pose
# Thumb Yaw/Roll used ONLY during the approach+descent (fingers/thumb open,
# no curl) -- found by sweeping the full Yaw/Roll range with the thumb fully
# straight (Pitch=Flexor=DIP=0) and picking whichever minimizes how far the
# tip extends past the palm's own body in the palm-normal direction. Can't
# be reduced to zero with this hand's geometry (confirmed by the sweep: the
# straight thumb is simply longer than the palm is thick -- best case still
# protrudes ~9.4cm vs the palm's own ~5.1cm), but this is the least-bad
# achievable position, used only during the open-handed approach/descent.
PRE_DESCENT_THUMB_YAW_RAD = -0.5
PRE_DESCENT_THUMB_ROLL_RAD = -0.34
PRE_DESCENT_THUMB_CURL = 0.95  # the curl the thumb reaches by the END of DESCEND (see PRESHAPE_THUMB_CURL below for where it STARTS). Must match finger_alignment_node.py's constant of the same name -- close_hand()'s thumb_curl_reversed logic starts from this value. RAISED 0.65 -> 0.95 (2026-10-16): 0.65 was tuned against the OLD GRASP_POINT_IN_HAND_M; after relocating the grasp point, 0.65 produced a hard live thumb-apple collision during descent (confirmed: effort -21.94, apple displaced 22mm). FK segment-distance sweep showed 0.65 was only ~0.056m clear (apple radius 0.04m) at the new point -- not enough real margin. 0.95 gives ~0.068m. See finger_alignment_node.py's close_hand() thumb_curl_reversed comment for how contact is still achieved: the thumb ramps DOWN from this resting value during close (not up), since FK shows its closest approach to the apple happens at a MODERATE curl (~0.3-0.4), not the max.

# ---------------------------------------------------------------------------
# PRESHAPE / DESCEND config (2026-10-16, user request: "thumb stays OPEN and
# OPPOSED until the hand is in position, then all digits close together" --
# one config block, not scattered magic numbers).
#
# Investigated first (per user's explicit request) before changing anything:
# the thumb was NOT closing due to a bug -- open_hand_preshape() was setting
# it to a heavily-curled rest pose (PRE_DESCENT_THUMB_CURL) in ONE command
# before the arm ever starts moving, and holding it there (no hand joints are
# ever included in the descent's own trajectory -- publish_multi_point_
# trajectory only ever sets ARM_JOINTS). That curl was a deliberate,
# twice-independently-confirmed-live safety fix: an extended thumb hits the
# apple during the final approach at this hand's geometry (collision #1,
# 2026-10-09, effort 37.4; collision #2, 2026-10-16 after the grasp point
# moved, effort -21.94). A fully-open thumb through the WHOLE descent isn't
# safe here.
#
# ATTEMPTED resolution (same day, reverted): start PRESHAPE with the thumb
# "mostly extended" and ramp the curl UP defensively DURING descend,
# synchronized with the arm's DESCENT_WAYPOINTS, plus a slight finger
# pre-curl so fingers come down around the apple's sides. FK-verified
# (combined_descend_close_check.py) clear at every intermediate point (worst
# case 0.068m) -- but LIVE testing showed a WORSE collision than the
# previous constant-curl approach: effort spikes on the thumb AND Pinky/
# Middle/Index during descent (fingers were independently confirmed safe
# fully straight; the slight pre-curl is the prime suspect). This offline FK
# model (zero-thickness line segments) has now underestimated real mesh
# collision risk multiple times this session -- not trustworthy enough for
# margins this fine. REVERTED: open_hand_preshape() holds the thumb at
# PRE_DESCENT_THUMB_CURL constant through PRESHAPE+DESCEND (no ramp, see its
# docstring), fingers stay fully straight (0.0). These two constants are
# kept (unused in the current flow) so a future, more conservative attempt
# at "thumb open longer" doesn't have to re-derive this from scratch.
PRESHAPE_THUMB_CURL = 0.15
PRESHAPE_FINGER_CURL = 0.2
HAND_SETTLE_TOLERANCE_RAD = 0.05  # PRESHAPE settle check: how close hand joints must get to their targets
HAND_SETTLE_TIMEOUT_SEC = 3.0     # PRESHAPE settle check: give up and proceed anyway after this long
OBSERVE_DIR_IK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "approach_observations")
# ikpy's end frame is at the same point as dexhand_base_link (measured via TF),
# rotated by a fixed 180 deg about X.
HAND_FRAME_FROM_IK_FRAME = np.diag([1.0, -1.0, -1.0])

# Reasonable top-down-reach seed for ikpy's numerical solver -- fully
# constraining both position AND orientation (6 DOF on a 6-joint arm) leaves
# it zero spare freedom, so an unseeded solve is more likely to converge to a
# bad local minimum (confirmed live: our earlier same-night attempt did
# exactly that, converging to the hand facing away). Seeding near a plausible
# top-down configuration instead of solver-default/random.
IK_SEED_ARM_ANGLES = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

MOVE_DURATION_SEC = 4.0
SETTLE_MARGIN_SEC = 1.0
VERIFY_TIMEOUT_SEC = 120.0  # CPU-only VLM inference can take ~60-90s per call

# Confirm the arm has ACTUALLY reached the target via real forward-kinematics
# feedback from /joint_states before curling any fingers, instead of blindly
# trusting MOVE_DURATION_SEC+SETTLE_MARGIN_SEC as a fixed timer. Confirmed
# live 2026-09-16: for a large reconfiguration the arm can still be mid-
# motion when that fixed timer fires, so Pitch was curling in while the hand
# was still traveling -- itself a likely contributor to pushing the object.
SETTLE_POSITION_TOLERANCE_M = 0.01  # was 0.03 -- confirmed live the 2.9cm slop here was landing mostly along the hand's reach (Z) axis, shortchanging the scoop depth
SETTLE_POLL_INTERVAL_SEC = 0.3
SETTLE_MAX_WAIT_SEC = 20.0  # more time to actually converge to the tighter tolerance above

ALIGNMENT_TOLERANCE_PX = 25.0
MAX_CORRECTION_ATTEMPTS = 4
MAX_INCONCLUSIVE_RETRIES = 3
CORRECTION_GAIN = 0.6
MIN_SECONDS_BETWEEN_NEW_TARGETS = 5.0

TARGET_KEYWORDS = ["apple"]


class PickState(Enum):
    IDLE = auto()
    MOVING = auto()
    AWAITING_ALIGNMENT_CHECK = auto()


class VisionIKOverheadClosedLoopNode(Node):
    def __init__(self):
        # use_sim_time -- without it every create_timer() delay below runs on
        # the WALL clock while trajectory durations (MOVE_DURATION_SEC etc.)
        # execute in SIM time. Under any real-time-factor below 1.0 (this sim
        # has been visibly heavy all night), wall-clock timers fire BEFORE
        # the sim has actually finished the previous move -- confirmed live
        # 2026-09-16 as the real cause of several "next stage starts before
        # the last one visibly finished" reports tonight. manual_target_node
        # .py already does this correctly; this file never did.
        super().__init__(
            "vision_ik_overhead_closed_loop_node",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base_link"])
        self.get_logger().info(f"ikpy chain loaded with {len(self.chain.links)} links.")

        self.active_mask = [False] * len(self.chain.links)
        for i in range(2, 8):
            if i < len(self.active_mask):
                self.active_mask[i] = True
        self.chain.active_links_mask = self.active_mask

        self.traj_pub = self.create_publisher(JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        self.hand_traj_pub = self.create_publisher(JointTrajectory, HAND_TRAJECTORY_TOPIC, 10)
        self.gripper_handoff_pub = self.create_publisher(String, GRIPPER_FRAGILITY_TOPIC, 10)
        self.alignment_request_pub = self.create_publisher(String, ALIGNMENT_REQUEST_TOPIC, 10)
        self.finger_align_trigger_pub = self.create_publisher(String, FINGER_ALIGN_TRIGGER_TOPIC, 10)

        self.create_subscription(String, FRAGILITY_TOPIC, self.on_fragility_msg, 10)
        self.create_subscription(String, ALIGNMENT_RESULT_TOPIC, self.on_alignment_result, 10)
        self.create_subscription(String, FINGER_SEQUENCE_DONE_TOPIC, self.on_finger_sequence_done, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)

        self.state = PickState.IDLE
        self.current_object_name = None
        self.current_target_base = None
        self.locked_target_px = None
        self.current_finger_targets = None
        self.current_real_size_m = None
        self.recently_grasped = []
        self.correction_attempts = 0
        self.inconclusive_retries = 0
        self.target_is_depth_precise = False
        self.latest_joint_positions = {}
        self._settle_target_base = None
        self._settle_wait_started = 0.0
        self._pending_timer = None
        self._last_new_target_time = 0.0
        self._grasp_correction = np.zeros(3)
        self._correction_count = 0
        self._last_settle_pos = None
        self._settle_callback = None
        self._approach_fk = None
        self._approach_angles = None
        self._apple_pos_before_motion = None
        self._apple_pos_start_of_descent = None
        self._descent_monitor_timer = None
        self._descent_contact_seen = {}
        self._preshape_wait_started = 0.0
        self._preshape_target_base = None
        self._preshape_targets = {}
        self.latest_detected_base = None  # continuously updated regardless of state (2026-10-09)
        self.latest_hand_effort = {}
        os.makedirs(OBSERVE_DIR_IK, exist_ok=True)
        self.run_log_path = os.path.join(OBSERVE_DIR_IK, time.strftime("%Y%m%d_%H%M%S") + ".log")

        self.get_logger().info("Vision IK closed-loop overhead node ready.")
        self.get_logger().info(f"Logging approach/descent observations to {self.run_log_path}")

    def on_fragility_msg(self, msg: String):
        try:
            analysis = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        obj = analysis.get("object_name", "unknown object")
        if obj != "none visible" and any(keyword in obj.lower() for keyword in TARGET_KEYWORDS):
            # ALWAYS cache the latest detected position, even mid-motion --
            # used by _on_approach_settled to re-read the apple's current
            # position before descending, per user request (2026-10-09):
            # "target that, not the spawn position."
            fresh = self.resolve_target_base(analysis, analysis.get("bbox_center_x", IMAGE_WIDTH_PX / 2),
                                             analysis.get("bbox_center_y", IMAGE_HEIGHT_PX / 2))
            if fresh is not None:
                self.latest_detected_base = fresh

        if self.state != PickState.IDLE:
            return

        if obj == "none visible":
            return

        if not any(keyword in obj.lower() for keyword in TARGET_KEYWORDS):
            self.get_logger().info(f"Ignoring detection '{obj}' -- doesn't match target keywords {TARGET_KEYWORDS}.")
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_new_target_time < MIN_SECONDS_BETWEEN_NEW_TARGETS:
            return
        self._last_new_target_time = now

        px = analysis.get("bbox_center_x", IMAGE_WIDTH_PX / 2)
        py = analysis.get("bbox_center_y", IMAGE_HEIGHT_PX / 2)

        if self._is_recently_grasped(px, py):
            self.get_logger().info(f"Ignoring detection near ({px}, {py}) -- recently grasped, still in cooldown.")
            return

        target_base = self.resolve_target_base(analysis, px, py)
        if target_base is None:
            return

        self.target_is_depth_precise = (
            "base_link_x" in analysis and "base_link_y" in analysis and "base_link_z" in analysis
        )

        # GROUND_OFFSET_BELOW_APPLE_M's top-down "aim below the apple" scoop
        # (2026-10-05) is RETIRED now that the approach itself is a side/
        # handshake grasp (2026-10-09, see ORIENTATION_MODE/SIDE_GRASP_
        # TARGET_ORIENTATION above) -- the height is now controlled
        # precisely by GRASP_POINT_IN_HAND_M's own Y component (derived from
        # the real palm geometry), not by shifting the target. Shifting Z
        # here too would double-count the height correction.

        self.get_logger().info(
            f"New target '{obj}' -> base_link ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})"
            f"{' [depth-precise]' if self.target_is_depth_precise else ''}"
        )

        self.current_object_name = obj
        self.current_target_base = target_base
        self.locked_target_px = (px, py)
        self.current_finger_targets = analysis.get("finger_joint_targets")
        self.current_real_size_m = analysis.get("real_size_m")
        self.correction_attempts = 0
        self.inconclusive_retries = 0
        self._grasp_correction = np.zeros(3)
        self._correction_count = 0
        self._last_settle_pos = None
        self._apple_pos_before_motion = None
        self._apple_pos_start_of_descent = None
        self._observe("before_motion")
        self.open_hand_preshape()
        self._preshape_wait_started = self.get_clock().now().nanoseconds / 1e9
        self._preshape_target_base = target_base
        self._pending_timer = self.create_timer(SETTLE_POLL_INTERVAL_SEC, self._check_preshape_settled)

    def begin_pick(self, target_base):
        """Stage 1 of 2 (user request, 2026-10-09): move to a point
        APPROACH_HEIGHT_ABOVE_M directly above the apple FIRST, with no
        horizontal motion below that height -- the straight-down descent
        itself happens in _on_approach_settled, re-solved as several
        independent IK waypoints along a vertical line (not a 2-point
        joint-space interpolation, which doesn't guarantee a straight
        Cartesian path)."""
        approach_target = (target_base[0], target_base[1], target_base[2] + APPROACH_HEIGHT_ABOVE_M)
        approach_angles = self.solve_ik_hand_frame(approach_target)
        if approach_angles is None:
            self.get_logger().error("Approach IK failed; aborting this pick attempt.")
            self.reset_to_idle()
            return
        self.get_logger().info(f"Approach IK solution (arm joints): {approach_angles}")
        self._approach_angles = approach_angles
        self.publish_trajectory(approach_angles)
        self.state = PickState.MOVING

        ja = [0.0] * len(self.chain.links)
        for i, a in enumerate(approach_angles):
            ja[2 + i] = a
        self._settle_target_base = tuple(self.chain.forward_kinematics(ja)[:3, 3])
        self._settle_wait_started = self.get_clock().now().nanoseconds / 1e9
        self._settle_callback = self._on_approach_settled
        self._pending_timer = self.create_timer(SETTLE_POLL_INTERVAL_SEC, self._check_arm_settled)

    def _on_approach_settled(self):
        """DESCEND now ENDS here, at APPROACH_HEIGHT_ABOVE_M above the final
        grasp point (2026-10-16, user: "don't let the thumb close while
        descending, keep the palm open"). REMOVED the second multi-waypoint
        Cartesian descent that used to continue all the way to the final
        depth with a fast, OPEN-LOOP arm trajectory and a static/ramping
        hand posture -- that's what made collisions during this stage
        dangerous: no contact feedback until the damage was already done
        (confirmed live, repeatedly, across several curl schedules this
        session, including the one FK had predicted safe).

        The remaining descent to the actual grasp depth now happens inside
        finger_alignment_node.py's descend_and_close(), which moves in small
        steps under the SAME effort-threshold contact detection that
        close_hand() already uses safely for the digits -- so a thumb that's
        genuinely open the whole time (PRESHAPE curl, unchanged since this
        point) gets caught gently on first contact instead of swept into a
        hard, fast collision. This hands off earlier than before, but
        otherwise reuses the SAME handoff vision_ik would have sent once the
        old descent finished (see _on_settle_timer_fired)."""
        self._observe("approach_settled")
        hand_joint_names = [n for n in self.latest_joint_positions if n.startswith("R_")]
        hand_state = {n: round(self.latest_joint_positions[n], 3) for n in sorted(hand_joint_names)}
        self.get_logger().info(f"[phase] DESCEND complete (stopped at APPROACH_HEIGHT_ABOVE_M) -- thumb/finger joint states: {hand_state}")
        # Re-read the apple's CURRENT position (user request: "target that,
        # not the spawn position") -- self.latest_detected_base is kept
        # continuously fresh by on_fragility_msg regardless of state. This is
        # now the target finger_alignment_node.py's descend_and_close() will
        # aim for.
        descent_target = self.latest_detected_base or self.current_target_base
        self.get_logger().info(
            f"Handing off to finger_alignment_node.py for the final approach, targeting "
            f"{tuple(round(v,3) for v in descent_target)} (original was {tuple(round(v,3) for v in self.current_target_base)})."
        )
        self.current_target_base = descent_target
        self._on_settle_timer_fired()

    def _monitor_descent_contact(self):
        for name, effort in self.latest_hand_effort.items():
            if not name.startswith("R_"):
                continue
            if abs(effort) > 0.5 and name not in self._descent_contact_seen:
                self._descent_contact_seen[name] = round(effort, 2)
                self.get_logger().warning(f"[descent contact] {name} effort={effort:.2f} DURING descent (not at a checkpoint).")
                with open(self.run_log_path, "a") as f:
                    f.write(f"[descent_contact] {name} effort={effort:.2f}\n")

    def _on_descent_settled(self):
        if self._descent_monitor_timer is not None:
            self.destroy_timer(self._descent_monitor_timer)
            self._descent_monitor_timer = None
        if self._descent_contact_seen:
            self._observe("descent_settled")
            self.get_logger().info(f"Contacts detected during descent: {self._descent_contact_seen}")
        else:
            self._observe("descent_settled")
        thumb_now = {
            j: round(self.latest_joint_positions.get(j, float("nan")), 3)
            for j in ("R_Thumb_Pitch", "R_Thumb_Flexor", "R_Thumb_DIP", "R_Thumb_Yaw", "R_Thumb_Roll")
        }
        self.get_logger().info(f"[phase] DESCEND complete -- thumb joint positions: {thumb_now} (target curl {PRE_DESCENT_THUMB_CURL})")
        self._log_arm_link_heights()
        self._verify_grasp_geometry(self.current_target_base)
        self._on_settle_timer_fired()

    def _log_arm_link_heights(self):
        """User-requested verification (2026-10-09): print the real height
        of every arm link at the grasp pose, from the ACTUAL joint_states,
        not the commanded solution -- confirms only the fingers end up near
        the ground, not the elbow/forearm."""
        for name in ARM_JOINTS:
            if name not in self.latest_joint_positions:
                self.get_logger().warning("No joint_states yet; can't verify link heights.")
                return
        ja = [0.0] * len(self.chain.links)
        for i, name in enumerate(ARM_JOINTS):
            ja[2 + i] = self.latest_joint_positions[name]
        full_fk = self.chain.forward_kinematics(ja, full_kinematics=True)
        self.get_logger().info("Arm link heights above ground at grasp pose:")
        for i in range(len(self.chain.links)):
            h = full_fk[i][2, 3] + BASE_LINK_HEIGHT_ABOVE_GROUND_M
            flag = " <-- BELOW SAFE HEIGHT" if (self.chain.links[i].name in ARM_SAFETY_CHECK_LINKS and h < SAFE_ARM_HEIGHT_ABOVE_GROUND_M) else ""
            self.get_logger().info(f"  {self.chain.links[i].name}: {h:.3f}m{flag}")

    def _check_arm_settled(self):
        """Poll real forward-kinematics from /joint_states until the gripper
        has actually reached target_base, instead of trusting a fixed timer
        that can fire before a large reconfiguration finishes moving."""
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self._settle_wait_started
        gripper_pos = self._compute_gripper_fk_position()

        callback = self._settle_callback or self._on_settle_timer_fired
        if gripper_pos is not None:
            dist = math.dist(gripper_pos, self._settle_target_base)
            if dist <= SETTLE_POSITION_TOLERANCE_M:
                self._cancel_pending_timer()
                self.get_logger().info(
                    f"Arm confirmed at target (FK offset {dist:.3f}m, {elapsed:.1f}s)."
                )
                callback()
                return

        if elapsed >= SETTLE_MAX_WAIT_SEC:
            self._cancel_pending_timer()
            self.get_logger().warn(
                f"Arm did not confirm reaching target within {SETTLE_MAX_WAIT_SEC:.0f}s; "
                "proceeding anyway."
            )
            callback()

    def _check_preshape_settled(self):
        """PRESHAPE settle check (2026-10-16, user request: tolerance check,
        not a fixed sleep): poll real /joint_states until the thumb's curl
        joints and Yaw/Roll actually reach their PRESHAPE targets, instead of
        assuming the 0.3s trajectory duration was enough."""
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self._preshape_wait_started
        check_joints = {
            "R_Thumb_Pitch": 0.0, "R_Thumb_Flexor": 0.0,
            "R_Thumb_DIP": 0.0, "R_Thumb_Yaw": PRE_DESCENT_THUMB_YAW_RAD,
            "R_Thumb_Roll": PRE_DESCENT_THUMB_ROLL_RAD,
        }
        have_all = all(j in self.latest_joint_positions for j in check_joints)
        if have_all:
            worst = max(abs(self.latest_joint_positions[j] - target) for j, target in check_joints.items())
            if worst <= HAND_SETTLE_TOLERANCE_RAD:
                self._cancel_pending_timer()
                self.get_logger().info(f"[phase] PRESHAPE settled (worst joint offset {worst:.3f}rad, {elapsed:.1f}s).")
                self.begin_pick(self._preshape_target_base)
                return
        if elapsed >= HAND_SETTLE_TIMEOUT_SEC:
            self._cancel_pending_timer()
            self.get_logger().warning(f"PRESHAPE did not confirm settling within {HAND_SETTLE_TIMEOUT_SEC:.0f}s; proceeding anyway.")
            self.begin_pick(self._preshape_target_base)

    def _compute_gripper_fk_position(self):
        for name in ARM_JOINTS:
            if name not in self.latest_joint_positions:
                return None
        joint_array = [0.0] * len(self.chain.links)
        for i, name in enumerate(ARM_JOINTS):
            joint_array[2 + i] = self.latest_joint_positions[name]
        fk_transform = self.chain.forward_kinematics(joint_array)
        position = fk_transform[:3, 3]
        return (float(position[0]), float(position[1]), float(position[2]))

    def on_joint_states(self, msg: JointState):
        for name, position, effort in zip(msg.name, msg.position, msg.effort):
            self.latest_joint_positions[name] = position
            self.latest_hand_effort[name] = effort

    def _observe(self, label):
        """Log apple's live GROUND-TRUTH world position (via Gazebo, not the
        detector -- so this is an independent check, not trusting the same
        pipeline being tested) plus any hand joint showing real effort
        (contact) while nothing is commanding it to move. Answers "did the
        apple move / did anything touch it before the fingers close"."""
        try:
            out = subprocess.run(
                "ign topic -e -t /world/apple_world/pose/info -n 1",
                shell=True, capture_output=True, text=True, timeout=10,
            ).stdout
            import re
            m = re.search(r'name: "apple_05".*?position \{(.*?)\}', out, re.S)
            vals = dict(re.findall(r'([xyz]): ([-\d.e]+)', m.group(1))) if m else {}
            apple_pos = [round(float(vals.get(k, 0.0)), 4) for k in "xyz"]
        except Exception as e:
            apple_pos = f"ERROR: {e}"

        hand_joints = [j for j in self.latest_hand_effort if j.startswith("R_")]
        contacts = {j: round(self.latest_hand_effort[j], 2) for j in hand_joints if abs(self.latest_hand_effort[j]) > 0.3}
        line = f"[{label}] apple_ground_truth={apple_pos}  unexpected_hand_contact={contacts or 'none'}\n"
        with open(self.run_log_path, "a") as f:
            f.write(line)
        self.get_logger().info(f"[observe] {line.strip()}")
        # User tolerance (2026-10-14): apple must move < 5mm before closing.
        # Checked over BOTH segments separately -- before_motion->
        # approach_settled (start of descent) and approach_settled->
        # descent_settled (the descent itself) -- so a failure points at
        # which phase actually caused it, not just the total.
        if label == "before_motion" and isinstance(apple_pos, list):
            self._apple_pos_before_motion = apple_pos
            self._apple_pos_start_of_descent = None
        elif label == "approach_settled" and isinstance(apple_pos, list):
            self._apple_pos_start_of_descent = apple_pos
            if self._apple_pos_before_motion is not None:
                disp = np.linalg.norm(np.array(apple_pos) - np.array(self._apple_pos_before_motion))
                self.get_logger().info(
                    f"[verify] apple displacement before_motion -> start_of_descent = {disp*1000:.1f}mm "
                    f"(target <= 5mm): {'PASS' if disp <= 0.005 else 'FAIL'}"
                )
        elif label == "descent_settled" and isinstance(apple_pos, list) and self._apple_pos_start_of_descent is not None:
            disp = np.linalg.norm(np.array(apple_pos) - np.array(self._apple_pos_start_of_descent))
            verdict = "PASS" if disp <= 0.005 else "FAIL"
            self.get_logger().info(
                f"[verify] apple displacement start_of_descent -> descent_settled (the descent itself) = {disp*1000:.1f}mm "
                f"(target <= 5mm): {verdict}"
                + (f"  -- contact during this window: {contacts}" if disp > 0.005 and contacts else "")
            )
        return apple_pos

    def _hand_local_fk(self):
        """Small self-contained FK for the thumb tip and 4-finger centroid in
        the hand's own local frame, using the SAME joint values
        finger_alignment_node.py actually commands at the align/close-ready
        posture (duplicated here deliberately -- these two processes don't
        share state -- must be kept in sync with that file's THUMB_YAW_RAD /
        THUMB_ROLL_RAD / GRIP_FAN_OUTER_RAD if those ever change)."""
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
            return np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s], [y*x*C+z*s, y*y*C+c, y*z*C-x*s], [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])

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

        # This verification runs right after descent, BEFORE close_hand's
        # ramp (2026-10-14) -- so the thumb is still at its PRE_DESCENT pose
        # (abducted + curled), not the opposed/straight close-ready one.
        # Using the real current values, not the close-ready ones, so this
        # check reflects where the thumb actually is at this moment.
        GRIP_FAN_OUTER_RAD = 0.28
        unit = GRIP_FAN_OUTER_RAD / 3.0
        thumb_tip = tip(["R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch", "R_Thumb_Flexor", "R_Thumb_DIP"],
                        {"R_Thumb_Yaw": PRE_DESCENT_THUMB_YAW_RAD, "R_Thumb_Roll": PRE_DESCENT_THUMB_ROLL_RAD,
                         "R_Thumb_Pitch": PRE_DESCENT_THUMB_CURL, "R_Thumb_Flexor": PRE_DESCENT_THUMB_CURL, "R_Thumb_DIP": PRE_DESCENT_THUMB_CURL})
        finger_tips = []
        for name, yaw in (("Index", -1.5*unit), ("Middle", -0.5*unit), ("Ring", 0.5*unit), ("Pinky", 1.5*unit)):
            chain = [f"R_{name}_Yaw", f"R_{name}_Pitch", f"R_{name}_Flexor", f"R_{name}_DIP"]
            finger_tips.append(tip(chain, {f"R_{name}_Yaw": yaw}))
        finger_centroid = np.mean(finger_tips, axis=0)
        palm_center_local = np.array([0.0166, -0.0042, 0.0407])  # STL bounding-box center, measured earlier
        return thumb_tip, finger_centroid, palm_center_local

    def _verify_grasp_geometry(self, apple_base):
        """User-requested checks (2026-10-14): grasp point (the point under
        the base of the fingers, GRASP_POINT_IN_HAND_M -- not the geometric
        palm center) vs apple centroid error, and open-hand span vs apple
        diameter, both printed with real numbers."""
        for name in ARM_JOINTS:
            if name not in self.latest_joint_positions:
                return
        ja = [0.0] * len(self.chain.links)
        for i, name in enumerate(ARM_JOINTS):
            ja[2 + i] = self.latest_joint_positions[name]
        fk = self.chain.forward_kinematics(ja)
        r_hand = fk[:3, :3] @ HAND_FRAME_FROM_IK_FRAME
        hand_pos = fk[:3, 3]

        thumb_tip_local, finger_centroid_local, palm_center_local = self._hand_local_fk()
        thumb_tip_world = hand_pos + r_hand @ thumb_tip_local
        finger_centroid_world = hand_pos + r_hand @ finger_centroid_local
        grasp_point_world = hand_pos + r_hand @ np.array(GRASP_POINT_IN_HAND_M)

        span = np.linalg.norm(thumb_tip_world - finger_centroid_world)
        apple_diam = 2 * APPLE_HALF_HEIGHT_M
        self.get_logger().info(
            f"[verify] open-hand span (thumb tip to finger centroid, straight) = {span:.3f}m, "
            f"apple diameter = {apple_diam:.3f}m -- span is measured at the OPEN pose, contact actually "
            f"happens partway through the close sweep (see GRASP_POINT_IN_HAND_M derivation notes)"
        )
        apple = np.array(apple_base)
        xy_err = np.linalg.norm(grasp_point_world[:2] - apple[:2])
        self.get_logger().info(
            f"[verify] grasp point world = ({grasp_point_world[0]:.3f}, {grasp_point_world[1]:.3f}, {grasp_point_world[2]:.3f})  "
            f"apple center xy = ({apple[0]:.3f}, {apple[1]:.3f})  "
            f"xy error = {xy_err*1000:.1f}mm (target <= 10mm): {'PASS' if xy_err <= 0.01 else 'FAIL'}"
        )

        # Item 3 (user, 2026-10-14): neither the (still-abducted/curled)
        # thumb nor the open fingers should overlap the apple's volume at
        # this pose -- apple radius + 1cm margin, checked against each tip.
        apple_r = APPLE_HALF_HEIGHT_M + 0.01
        thumb_gap = np.linalg.norm(thumb_tip_world - apple)
        finger_gap = np.linalg.norm(finger_centroid_world - apple)
        self.get_logger().info(
            f"[verify] thumb tip to apple center = {thumb_gap:.3f}m, finger centroid to apple center = {finger_gap:.3f}m "
            f"(need >= {apple_r:.3f}m to not overlap): "
            f"thumb {'OK' if thumb_gap >= apple_r else 'OVERLAP'}, fingers {'OK' if finger_gap >= apple_r else 'OVERLAP'}"
        )

    def _on_settle_timer_fired(self):
        self._cancel_pending_timer()
        if self.target_is_depth_precise:
            # Skip the VLM alignment check for depth-precise targets -- in
            # every test tonight it never once returned a usable result
            # (qwen2.5vl:3b can't reliably spot this hand from directly
            # overhead) and cost ~3.5 minutes per attempt for nothing, with
            # the hand sitting near the object that whole time.
            #
            # This node's job stops here: the arm is CONFIRMED (via FK
            # feedback, see _check_arm_settled) to have actually reached the
            # target with the hand still fully open (no pre-curl, see
            # open_hand_preshape). Hand off to finger_alignment_node.py for
            # straight-finger (no curl) positioning around the object -- a
            # separate, dedicated node, not adaptive_grasp_controller.py.
            handoff = {
                "object_name": self.current_object_name,
                "base_link_x": self.current_target_base[0],
                "base_link_y": self.current_target_base[1],
                "base_link_z": self.current_target_base[2],
                "real_size_m": self.current_real_size_m,
            }
            msg = String()
            msg.data = json.dumps(handoff)
            self.finger_align_trigger_pub.publish(msg)
            self.get_logger().info(
                f"Arm confirmed settled -- handed off to finger_alignment_node.py "
                f"for '{self.current_object_name}'."
            )
            # Stay MOVING (not idle) until finger_alignment_node.py's own
            # align->close->lift sequence is done -- confirmed live
            # 2026-09-16: without this, on_fragility_msg's IDLE-only guard
            # let a SECOND pick attempt start (new open_hand_preshape, which
            # UNCURLS the thumb) while the first was still mid-lift on the
            # SAME physical hand, and the uncurling thumb swept the object
            # away. finger_alignment_node.py reports back on
            # FINGER_SEQUENCE_DONE_TOPIC when it's actually finished.
            self._pending_timer = self.create_timer(
                FINGER_SEQUENCE_TIMEOUT_SEC, self._on_finger_sequence_timeout
            )
        else:
            self.request_alignment_check()

    def _on_finger_sequence_timeout(self):
        self._cancel_pending_timer()
        self.get_logger().warn(
            "No finger-sequence-done report in time; resetting to idle anyway."
        )
        self._mark_recently_grasped(*self.locked_target_px)
        self.reset_to_idle()

    def on_finger_sequence_done(self, msg: String):
        if self.state == PickState.IDLE:
            return
        self._cancel_pending_timer()
        self.get_logger().info("finger_alignment_node.py reported its sequence done.")
        self._mark_recently_grasped(*self.locked_target_px)
        self.reset_to_idle()

    def request_alignment_check(self):
        if self.current_object_name is None:
            self.reset_to_idle()
            return

        request = {
            "object_name": self.current_object_name,
            "check": "gripper_alignment",
            "target_pixel_hint": list(self.locked_target_px),
        }
        msg = String()
        msg.data = json.dumps(request)
        self.alignment_request_pub.publish(msg)

        self.state = PickState.AWAITING_ALIGNMENT_CHECK
        self.get_logger().info(
            f"Requested alignment check for '{self.current_object_name}' "
            f"(attempt {self.correction_attempts + 1}/{MAX_CORRECTION_ATTEMPTS})."
        )

        self._pending_timer = self.create_timer(VERIFY_TIMEOUT_SEC, self._on_verify_timeout)

    def _on_verify_timeout(self):
        self._cancel_pending_timer()
        if self.state == PickState.AWAITING_ALIGNMENT_CHECK:
            self.get_logger().warn("No alignment check result in time; giving up on this correction cycle.")
            self.reset_to_idle()

    def on_alignment_result(self, msg: String):
        if self.state != PickState.AWAITING_ALIGNMENT_CHECK:
            return

        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if result.get("object_name") != self.current_object_name:
            return

        self._cancel_pending_timer()

        gripper_visible = bool(result.get("gripper_visible", True))
        target_visible = bool(result.get("target_visible", True))

        if not gripper_visible or not target_visible:
            self.inconclusive_retries += 1
            if self.inconclusive_retries >= MAX_INCONCLUSIVE_RETRIES:
                if self.target_is_depth_precise:
                    # The VLM alignment check can't get a conclusive read (e.g. its
                    # small model can't reliably spot the gripper from directly
                    # overhead), but this target came from a real depth-camera
                    # measurement already verified accurate to ~3cm (well inside
                    # the apple's own ~4cm radius) -- trust that placement and
                    # grasp rather than aborting empty-handed on an unrelated
                    # vision-language limitation.
                    self.get_logger().warn(
                        f"Alignment check inconclusive after {self.inconclusive_retries} attempts, but "
                        f"'{self.current_object_name}' came from a depth-precise detection -- "
                        f"grasping at the measured position instead of giving up."
                    )
                    self.execute_grasp()
                    self._mark_recently_grasped(*self.locked_target_px)
                    self.reset_to_idle()
                    return

                self.get_logger().warn(
                    f"Gave up on alignment check for '{self.current_object_name}': "
                    f"gripper/target not visible after {self.inconclusive_retries} attempts."
                )
                self.reset_to_idle()
                return

            self.get_logger().warn(
                f"Alignment check inconclusive (gripper_visible={gripper_visible}, "
                f"target_visible={target_visible}). Retrying ({self.inconclusive_retries}/{MAX_INCONCLUSIVE_RETRIES})."
            )
            self._pending_timer = self.create_timer(SETTLE_MARGIN_SEC, self._on_settle_timer_fired)
            self.state = PickState.MOVING
            return

        aligned = bool(result.get("aligned", False))
        dx_px = float(result.get("offset_x_px", 0.0))
        dy_px = float(result.get("offset_y_px", 0.0))
        offset_mag_px = math.hypot(dx_px, dy_px)

        if aligned or offset_mag_px <= ALIGNMENT_TOLERANCE_PX:
            self.get_logger().info(
                f"Gripper aligned with '{self.current_object_name}' (offset {offset_mag_px:.1f}px). Closing hand."
            )
            self.execute_grasp()
            self._mark_recently_grasped(*self.locked_target_px)
            self.reset_to_idle()
            return

        self.correction_attempts += 1
        if self.correction_attempts >= MAX_CORRECTION_ATTEMPTS:
            self.get_logger().warn(
                f"Max correction attempts reached (last offset {offset_mag_px:.1f}px). Stopping without full alignment."
            )
            self.reset_to_idle()
            return

        self.inconclusive_retries = 0
        target_px = result.get("target_px")
        if target_px and len(target_px) == 2:
            self.locked_target_px = (float(target_px[0]), float(target_px[1]))

        correction_base = self.pixel_offset_to_base_correction(dx_px, dy_px)
        corrected_target = (
            self.current_target_base[0] + correction_base[0],
            self.current_target_base[1] + correction_base[1],
            self.current_target_base[2] + correction_base[2],
        )
        self.current_target_base = corrected_target

        self.get_logger().info(
            f"Misaligned by {offset_mag_px:.1f}px. Correcting -> "
            f"({corrected_target[0]:.3f}, {corrected_target[1]:.3f}, {corrected_target[2]:.3f})."
        )
        self.move_to(corrected_target, then_verify=True)

    def resolve_target_base(self, analysis, px, py):
        """Prefer a real, depth-measured 3D position when the detector
        supplied one (yolo_depth_detector.py publishes base_link_x/y/z from
        an actual depth-camera reading). This is strictly more precise than
        the bbox-center-plus-assumed-height projection below, since it
        doesn't need to guess the object's height off the ground. Fall back
        to the pixel-projection estimate only for detectors that can't see
        depth (apple_color_detector.py, vlm_overhead_node.py)."""
        if "base_link_x" in analysis and "base_link_y" in analysis and "base_link_z" in analysis:
            try:
                return (
                    float(analysis["base_link_x"]),
                    float(analysis["base_link_y"]),
                    float(analysis["base_link_z"]),
                )
            except (TypeError, ValueError):
                pass  # malformed fields -- fall through to the pixel estimate

        world_point = self.pixel_to_world(px, py)
        return self.transform_world_to_base(world_point)

    def pixel_offset_to_base_correction(self, dx_px, dy_px):
        camera_height_above_apples = CAMERA_WORLD_Z - APPLE_WORLD_Z
        m_per_px = camera_height_above_apples / FOCAL_LENGTH_PX
        d_world_x = V_SIGN * (dy_px * m_per_px) * CORRECTION_GAIN
        d_world_y = U_SIGN * (dx_px * m_per_px) * CORRECTION_GAIN
        return (d_world_x, d_world_y, 0.0)

    def pixel_to_world(self, px, py):
        camera_height_above_apples = CAMERA_WORLD_Z - APPLE_WORLD_Z
        m_per_px = camera_height_above_apples / FOCAL_LENGTH_PX
        dx_px = px - (IMAGE_WIDTH_PX / 2.0)
        dy_px = py - (IMAGE_HEIGHT_PX / 2.0)
        world_x = CAMERA_WORLD_X + V_SIGN * (dy_px * m_per_px)
        world_y = CAMERA_WORLD_Y + U_SIGN * (dx_px * m_per_px)
        world_z = APPLE_WORLD_Z
        return (world_x, world_y, world_z)

    def transform_world_to_base(self, world_point):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = WORLD_FRAME
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x = world_point[0]
            point_stamped.point.y = world_point[1]
            point_stamped.point.z = world_point[2]

            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, WORLD_FRAME, rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None

    def solve_ik_hand_frame(self, target_base, seed_override=None):
        """Place the wrist so the object lands at GRASP_POINT_IN_HAND_M in the
        HAND's own frame, whatever wrist roll IK picks: solve, read the actual
        hand rotation from FK, recompute the wrist target, repeat.

        seed_override (2026-10-09, user: "dont do so much turning"): when
        solving a SEQUENCE of nearby waypoints (the descent), re-searching
        all of IK_ALT_SEEDS at each one can land on a different solution
        branch from one waypoint to the next, producing a visible extra
        swing/rotation between them instead of a smooth path. Passing the
        PREVIOUS waypoint's own solution here skips that search and warm-
        starts from where the arm already is."""
        target = np.array(target_base, dtype=float)
        g = np.array(GRASP_POINT_IN_HAND_M)
        cmd = target - np.array(HAND_OFFSET_VECTOR_M)  # first guess only
        angles = None
        for it in range(6):
            angles = self.solve_ik(tuple(cmd), seed_override=seed_override)
            if angles is None:
                return None
            ja = [0.0] * len(self.chain.links)
            for i, a in enumerate(angles):
                ja[2 + i] = a
            fk = self.chain.forward_kinematics(ja)
            r_hand = fk[:3, :3] @ HAND_FRAME_FROM_IK_FRAME
            desired_wrist = target - r_hand @ g + self._grasp_correction
            err = desired_wrist - fk[:3, 3]
            if np.linalg.norm(err) < 0.002:
                break
            cmd = cmd + err
        self.get_logger().info(
            f"Hand-frame IK: grasp point {tuple(g)} m in hand frame, residual "
            f"{np.linalg.norm(err)*1000:.1f} mm after {it + 1} iterations, achieved wrist {np.round(fk[:3, 3], 3).tolist()}"
        )
        return angles

    def solve_ik(self, target_base, seed_override=None, _retry_pass=False):
        target_pos = np.array(target_base, dtype=float)
        if seed_override is not None:
            seeds = [seed_override]
        elif ORIENTATION_MODE == "all":
            target_orient = SIDE_GRASP_TARGET_ORIENTATION
            seeds = IK_ALT_SEEDS
        else:
            target_orient = np.array(TARGET_APPROACH_DIRECTION)
            seeds = [IK_SEED_ARM_ANGLES]
        if seed_override is not None:
            target_orient = SIDE_GRASP_TARGET_ORIENTATION if ORIENTATION_MODE == "all" else np.array(TARGET_APPROACH_DIRECTION)

        best_angles, best_err = None, None            # best overall (fallback only)
        best_safe_angles, best_safe_err = None, None  # best among arm-safe candidates -- preferred
        for seed_vals in seeds:
            seed = [0.0] * len(self.chain.links)
            for i, val in enumerate(seed_vals):
                seed[2 + i] = val
            try:
                ik_solution = self.chain.inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_orient,
                    orientation_mode=ORIENTATION_MODE,
                    initial_position=seed,
                )
            except Exception as e:
                self.get_logger().error(f"IK solve failed for seed {seed_vals}: {e}")
                continue

            ja = [0.0] * len(self.chain.links)
            for i in range(6):
                ja[2 + i] = ik_solution[2 + i]
            fk = self.chain.forward_kinematics(ja)
            pos_err = np.linalg.norm(fk[:3, 3] - target_pos)
            if ORIENTATION_MODE == "all":
                orient_err = np.linalg.norm(fk[:3, :3] - target_orient)
            else:
                orient_err = 0.0
            err = pos_err + orient_err
            if best_err is None or err < best_err:
                best_err = err
                best_angles = [ik_solution[i] for i in range(2, 8)]

            # Arm-safety check (user request, 2026-10-09): reject -- don't
            # just deprioritize -- any candidate where the elbow or forearm
            # (wrist_1) link is below SAFE_ARM_HEIGHT_ABOVE_GROUND_M.
            # Confirmed live this was happening silently: most seeds here
            # converge to wrist_1 only 4cm off the floor even though their
            # position/orientation residual looks fine, so residual alone
            # can't be trusted to avoid it.
            full_fk = self.chain.forward_kinematics(ja, full_kinematics=True)
            link_heights = {
                self.chain.links[i].name: full_fk[i][2, 3] + BASE_LINK_HEIGHT_ABOVE_GROUND_M
                for i in range(len(self.chain.links))
            }
            arm_safe = all(link_heights.get(name, 0.0) >= SAFE_ARM_HEIGHT_ABOVE_GROUND_M for name in ARM_SAFETY_CHECK_LINKS)
            if arm_safe and (best_safe_err is None or err < best_safe_err):
                best_safe_err = err
                best_safe_angles = [ik_solution[i] for i in range(2, 8)]

            if len(seeds) > 1:
                heights_str = " ".join(f"{n}={link_heights.get(n, 0):.3f}m" for n in ARM_SAFETY_CHECK_LINKS)
                self.get_logger().info(
                    f"  seed {seed_vals}: pos_err={pos_err*1000:.1f}mm orient_err={orient_err:.3f} "
                    f"{heights_str} {'[SAFE]' if arm_safe else '[REJECTED: forearm too low]'}"
                )

        if best_safe_angles is not None:
            best_angles, best_err = best_safe_angles, best_safe_err
        elif seed_override is not None and not _retry_pass:
            # The continuity seed (previous waypoint's own solution) drifted
            # into an unsafe elbow-down config with no other candidate to
            # fall back on. Re-search the full seed list fresh instead of
            # accepting the unsafe pose -- confirmed live (2026-10-14) this
            # is what was letting the forearm settle below ground on the
            # final descent waypoint even though a safe solution existed.
            self.get_logger().warning(
                "Continuity seed produced an unsafe (forearm-too-low) solution -- "
                "re-searching the full seed list for a safe alternative."
            )
            retried = self.solve_ik(target_base, seed_override=None, _retry_pass=True)
            if retried is not None:
                return retried
            self.get_logger().warning(
                "No IK candidate kept the elbow/forearm above the safe height -- "
                "falling back to the best-residual solution anyway (this grasp attempt may hit the ground)."
            )
        elif best_angles is not None:
            self.get_logger().warning(
                "No IK candidate kept the elbow/forearm above the safe height -- "
                "falling back to the best-residual solution anyway (this grasp attempt may hit the ground)."
            )

        if best_angles is None:
            self.get_logger().error("IK solve failed for all seeds.")
            return None
        if len(seeds) > 1:
            self.get_logger().info(f"IK: best seed residual {best_err:.4f} (pos mm + orientation).")
        return best_angles

    def publish_trajectory(self, joint_angles):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(MOVE_DURATION_SEC)
        point.time_from_start.nanosec = int((MOVE_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.traj_pub.publish(traj)
        self.get_logger().info("Published trajectory.")

    def publish_multi_point_trajectory(self, waypoints):
        """N-point Cartesian-straight descent -- each waypoint was already
        independently IK-solved at the fresh apple x,y, so this just
        schedules them in time order, evenly spaced."""
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        n = len(waypoints)
        for idx, angles in enumerate(waypoints, start=1):
            t = MOVE_DURATION_SEC * idx / n
            point = JointTrajectoryPoint()
            point.positions = angles
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            traj.points.append(point)
        self.traj_pub.publish(traj)
        self.get_logger().info(f"Published {n}-waypoint straight-down descent trajectory.")

    def publish_thumb_curl_ramp(self, n_waypoints):
        """Synchronized DEFENSIVE thumb curl-up during DESCEND (2026-10-16):
        ramps Pitch/Flexor/DIP from PRESHAPE_THUMB_CURL (mostly extended,
        set by open_hand_preshape) to PRE_DESCENT_THUMB_CURL (the
        FK-verified safe curl at the final depth) in lockstep with the arm's
        own DESCENT_WAYPOINTS, same n and same MOVE_DURATION_SEC timebase as
        publish_multi_point_trajectory, so the thumb is never more extended
        than is safe for its current height -- see the PRESHAPE/DESCEND
        config block's comment for the FK verification this schedule is
        based on (combined_descend_close_check.py, worst-case clearance
        0.068m at any intermediate point)."""
        thumb_joints = ["R_Thumb_Pitch", "R_Thumb_Flexor", "R_Thumb_DIP"]
        traj = JointTrajectory()
        traj.joint_names = thumb_joints
        for k in range(1, n_waypoints + 1):
            frac = k / n_waypoints
            curl = PRESHAPE_THUMB_CURL + frac * (PRE_DESCENT_THUMB_CURL - PRESHAPE_THUMB_CURL)
            t = MOVE_DURATION_SEC * k / n_waypoints
            point = JointTrajectoryPoint()
            point.positions = [curl, curl, curl]
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            traj.points.append(point)
        self.hand_traj_pub.publish(traj)

    def publish_two_point_trajectory(self, approach_angles, final_angles):
        """Waypoint 1 (approach, above the apple) then waypoint 2 (final
        pre-close pose) in ONE trajectory message, so the controller
        interpolates smoothly through both instead of two separate moves
        racing each other."""
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        p1 = JointTrajectoryPoint()
        p1.positions = approach_angles
        t1 = MOVE_DURATION_SEC * 0.5
        p1.time_from_start.sec = int(t1)
        p1.time_from_start.nanosec = int((t1 % 1) * 1e9)
        p2 = JointTrajectoryPoint()
        p2.positions = final_angles
        p2.time_from_start.sec = int(MOVE_DURATION_SEC)
        p2.time_from_start.nanosec = int((MOVE_DURATION_SEC % 1) * 1e9)
        traj.points.append(p1)
        traj.points.append(p2)
        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"Published 2-waypoint trajectory (approach {APPROACH_HEIGHT_ABOVE_M}m above, then descend)."
        )

    def open_hand_preshape(self):
        """PRESHAPE phase (2026-10-16, user: "keep the thumb from closing and
        moving towards the palm, keep the palm open"). Thumb is now
        genuinely open (curl 0.0) and rotated into opposition (Yaw/Roll),
        held there through ALL of PRESHAPE and DESCEND -- safe now because
        DESCEND no longer drives all the way down to the final grasp depth
        with a fast, open-loop trajectory (see _on_approach_settled): it
        stops at APPROACH_HEIGHT_ABOVE_M, a height with no realistic contact
        risk regardless of curl. The final approach (where contact risk
        actually lives) now happens inside finger_alignment_node.py's
        descend_and_close(), under the same effort-threshold contact
        detection the digits already use safely, so an open thumb gets
        caught gently on first contact instead of swept into a fast
        collision -- this is what makes "stay open" safe here, where it
        wasn't safe for the fast pre-planned descent this session tried
        earlier.

        Only publishes CLOSING_JOINTS|PITCH_JOINTS plus thumb Yaw/Roll in
        ONE message -- confirmed live 2026-09-17: splitting a stationary
        joint from an actively-moving one across two trajectory messages let
        the stationary one visibly bulge mid-motion (interpolation
        artifact)."""
        if not self.current_finger_targets:
            return

        # RAW radians, sent DIRECTLY -- no normalized->radians conversion
        # (confirmed correct independently via FK + a bare topic-pub test,
        # 2026-09-17).
        targets = {j: 0.0 for j in (CLOSING_JOINTS | PITCH_JOINTS)}  # fingers AND thumb fully straight/open
        targets.update(PRE_DESCENT_FAN_RAD)
        targets["R_Thumb_Yaw"] = PRE_DESCENT_THUMB_YAW_RAD
        targets["R_Thumb_Roll"] = PRE_DESCENT_THUMB_ROLL_RAD
        traj = JointTrajectory()
        traj.joint_names = list(targets.keys())
        point = JointTrajectoryPoint()
        point.positions = list(targets.values())
        point.time_from_start.sec = int(HAND_OPEN_DURATION_SEC)
        point.time_from_start.nanosec = int((HAND_OPEN_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)
        self._preshape_targets = dict(targets)
        self.get_logger().info(
            f"[phase] PRESHAPE commanded: thumb curl=0.0 (open) "
            f"yaw={PRE_DESCENT_THUMB_YAW_RAD} roll={PRE_DESCENT_THUMB_ROLL_RAD}, finger curl=0.0 (straight)."
        )

    def execute_grasp(self):
        if not self.current_finger_targets:
            self.get_logger().warn("No finger_joint_targets available; skipping hand close.")
            return

        # Hand off to adaptive_grasp_controller.py rather than closing with
        # one fixed-force trajectory: it watches joint EFFORT in real time and
        # backs off the instant it detects deformation onset instead of
        # blindly driving every finger to a preset closure (which also used a
        # uniform (0, pi/2) placeholder for joint limits here, not the real
        # per-joint URDF limits it uses). It triggers automatically on a new
        # object_name on GRIPPER_FRAGILITY_TOPIC and takes the non-closing
        # (Pitch/Yaw/Roll) pre-shape values straight from finger_joint_targets.
        handoff = {
            "object_name": self.current_object_name,
            "finger_joint_targets": self.current_finger_targets,
        }
        msg = String()
        msg.data = json.dumps(handoff)
        self.gripper_handoff_pub.publish(msg)
        self.get_logger().info(
            f"Handed off grasp for '{self.current_object_name}' to adaptive_grasp_controller.py."
        )

    def _mark_recently_grasped(self, px, py):
        now = self.get_clock().now().nanoseconds / 1e9
        self.recently_grasped.append((px, py, now))

    def _is_recently_grasped(self, px, py):
        now = self.get_clock().now().nanoseconds / 1e9
        self.recently_grasped = [
            (gx, gy, t) for (gx, gy, t) in self.recently_grasped if now - t < GRASP_COOLDOWN_SEC
        ]
        for gx, gy, _ in self.recently_grasped:
            if math.hypot(px - gx, py - gy) <= GRASP_PROXIMITY_PX:
                return True
        return False

    def _cancel_pending_timer(self):
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self.destroy_timer(self._pending_timer)
            self._pending_timer = None

    def reset_to_idle(self):
        self._cancel_pending_timer()
        self.state = PickState.IDLE
        self.current_object_name = None
        self.current_target_base = None
        self.locked_target_px = None
        self.current_finger_targets = None
        self.correction_attempts = 0
        self.inconclusive_retries = 0
        self.target_is_depth_precise = False


def main(args=None):
    rclpy.init(args=args)
    node = VisionIKOverheadClosedLoopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
