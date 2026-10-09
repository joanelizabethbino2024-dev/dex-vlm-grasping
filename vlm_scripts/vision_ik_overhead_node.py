#!/usr/bin/env python3
"""
Vision-based IK Approach Node (Overhead Camera, Closed-Loop Version)
----------------------------------------------------------------------
Extends the original open-loop node with a verify-and-correct loop:

  1. Detect object (existing fragility/detection pipeline) -> compute target
     -> solve IK -> move.
  2. After the move settles, ask the VLM alignment pipeline to check how far
     the gripper is from the target in the overhead camera image.
  3. If the offset exceeds a tolerance, convert the pixel offset into a small
     world-frame correction and move again.
  4. Repeat until aligned or a max number of correction attempts is reached.

VLM INTEGRATION NOTE
---------------------
Like the original node, this assumes VLM inference happens in a separate
process/node (the same one that already publishes to
`/overhead_camera/fragility_analysis`), because that's the architecture the
original code implies (a JSON string arrives on a topic; nothing in this
node calls an API directly).

So alignment checks follow the same request/response topic convention:

  - This node publishes a small JSON request to
      /overhead_camera/alignment_check_request
    telling the external VLM process which object to look for and that it
    should locate the gripper too.

  - The external VLM process is expected to publish back on
      /overhead_camera/alignment_check_result
    JSON of the form:
      {
        "object_name": "apple_1",
        "aligned": false,
        "gripper_px": [px, py],
        "target_px": [px, py],
        "offset_x_px": dx,
        "offset_y_px": dy,
        "confidence": 0.91
      }

If your actual VLM integration is different (e.g. a ROS2 service, or a
direct HTTP/SDK call to a hosted vision model), replace the two methods
`request_alignment_check()` and `on_alignment_result()` accordingly -- the
state machine around them does not need to change.
"""
import os
import json
import math
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from ikpy.chain import Chain

FRAGILITY_TOPIC = "/overhead_camera/fragility_analysis"
ALIGNMENT_REQUEST_TOPIC = "/overhead_camera/alignment_check_request"
ALIGNMENT_RESULT_TOPIC = "/overhead_camera/alignment_check_result"
ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanded_robot.urdf")

WORLD_FRAME = "base_footprint"
BASE_FRAME = "base_link"

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# --- Hand / finger control ------------------------------------------------
# Confirmed correct (see vision_ik_overhead_closed_loop.py, added to the
# launch file after manual testing) -- this node was left on the old guess.
HAND_TRAJECTORY_TOPIC = "/dexhand_controller/joint_trajectory"

HAND_JOINTS = [
    "R_Thumb_Pitch", "R_Thumb_Roll", "R_Thumb_Yaw", "R_Thumb_Flexor", "R_Thumb_DIP",
    "R_Index_Pitch", "R_Index_Yaw", "R_Index_Flexor", "R_Index_DIP",
    "R_Middle_Pitch", "R_Middle_Yaw", "R_Middle_Flexor", "R_Middle_DIP",
    "R_Ring_Pitch", "R_Ring_Yaw", "R_Ring_Flexor", "R_Ring_DIP",
    "R_Pinky_Pitch", "R_Pinky_Yaw", "R_Pinky_Flexor", "R_Pinky_DIP",
]

# CONFIRM/ADJUST: real min/max radians per joint from your URDF. These are
# placeholders (0 to 90 degrees, 0.5 normalized = midpoint) -- normalized
# 0.0-1.0 values from the VLM are scaled into this range before publishing.
HAND_JOINT_LIMITS_RAD = {name: (0.0, math.pi / 2) for name in HAND_JOINTS}

HAND_CLOSE_DURATION_SEC = 2.0

GRASP_COOLDOWN_SEC = 90.0        # how long a just-grasped object is excluded from
                                  # being re-targeted as a new pick
GRASP_PROXIMITY_PX = 40.0        # a new detection within this many px of a recent
                                  # grasp is treated as "the same object, skip it"

CAMERA_WORLD_X = 1.125
CAMERA_WORLD_Y = 0.0
CAMERA_WORLD_Z = 2.0

IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
HORIZONTAL_FOV_RAD = 1.396
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)

APPLE_WORLD_Z = 0.05

U_SIGN = 1.0
V_SIGN = 1.0

# Top-down approach orientation for the IK solver. This tells ikpy to align
# the end effector's LOCAL Z axis with this direction vector in the base
# frame, rather than only solving for position (which was letting the
# solver reach the target from any arbitrary angle -- including sideways,
# as seen in practice).
#
# CONFIRMED via manual_target_node.py testing (see
# vision_ik_overhead_closed_loop.py): [0,0,1] gives a correct top-down
# approach for this robot's tool frame. The [0,0,-1] guess below was
# backward (hand faced away from the target) -- this node was left on it.
TARGET_APPROACH_DIRECTION = [0, 0, 1]
ORIENTATION_MODE = "Z"

MOVE_DURATION_SEC = 4.0
SETTLE_MARGIN_SEC = 1.0          # extra wait after a move before verifying
VERIFY_TIMEOUT_SEC = 120.0       # CPU-only VLM inference can take ~60-90s per call

ALIGNMENT_TOLERANCE_PX = 25.0    # gripper considered "aligned" within this. Loosened
                                  # from an earlier 8px -- a 3B VLM's pixel-position
                                  # estimates are noisy; 8px is tighter than the model
                                  # can reliably resolve, which was causing it to burn
                                  # through all correction attempts without ever
                                  # actually being wrong by much.
MAX_CORRECTION_ATTEMPTS = 4
MAX_INCONCLUSIVE_RETRIES = 3     # cap on "couldn't see gripper/target" retries
CORRECTION_GAIN = 0.6            # <1.0 damps corrections to avoid overshoot
MIN_SECONDS_BETWEEN_NEW_TARGETS = 5.0

# Only chase detections that plausibly ARE the intended target. Without this,
# the node will happily move toward anything the VLM claims to see -- including
# misidentified robot parts, mounts, or background objects.
TARGET_KEYWORDS = ["apple"]


class PickState(Enum):
    IDLE = auto()
    MOVING = auto()
    AWAITING_ALIGNMENT_CHECK = auto()


class VisionIKOverheadClosedLoopNode(Node):
    def __init__(self):
        super().__init__("vision_ik_overhead_closed_loop_node")

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
        self.alignment_request_pub = self.create_publisher(
            String, ALIGNMENT_REQUEST_TOPIC, 10
        )

        self.create_subscription(String, FRAGILITY_TOPIC, self.on_fragility_msg, 10)
        self.create_subscription(
            String, ALIGNMENT_RESULT_TOPIC, self.on_alignment_result, 10
        )

        # --- state machine bookkeeping ---
        self.state = PickState.IDLE
        self.current_object_name = None
        self.current_target_base = None      # (x, y, z) in base_link frame
        self.locked_target_px = None         # pixel position at lock time -- keeps
                                              # alignment checks anchored to one specific
                                              # object among several with the same name
        self.current_finger_targets = None   # normalized 0-1 targets from detection
        self.recently_grasped = []           # [(px, py, timestamp), ...]
        self.correction_attempts = 0
        self.inconclusive_retries = 0
        self._pending_timer = None
        self._last_new_target_time = 0.0

        self.get_logger().info("Vision IK closed-loop overhead node ready.")

    # ------------------------------------------------------------------
    # Step 1: initial detection -> move
    # ------------------------------------------------------------------
    def on_fragility_msg(self, msg: String):
        if self.state != PickState.IDLE:
            # Already mid-pick / mid-verification; ignore new detections
            # until the current attempt finishes.
            return

        try:
            analysis = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        obj = analysis.get("object_name", "unknown object")
        if obj == "none visible":
            return

        if not any(keyword in obj.lower() for keyword in TARGET_KEYWORDS):
            self.get_logger().info(
                f"Ignoring detection '{obj}' -- doesn't match target keywords "
                f"{TARGET_KEYWORDS}."
            )
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_new_target_time < MIN_SECONDS_BETWEEN_NEW_TARGETS:
            return
        self._last_new_target_time = now

        px = analysis.get("bbox_center_x", IMAGE_WIDTH_PX / 2)
        py = analysis.get("bbox_center_y", IMAGE_HEIGHT_PX / 2)

        if self._is_recently_grasped(px, py):
            self.get_logger().info(
                f"Ignoring detection near ({px}, {py}) -- matches a recently "
                f"grasped object, still within cooldown."
            )
            return

        world_point = self.pixel_to_world(px, py)
        target_base = self.transform_world_to_base(world_point)
        if target_base is None:
            return

        self.get_logger().info(
            f"New target '{obj}' -> base_link "
            f"({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})"
        )

        self.current_object_name = obj
        self.current_target_base = target_base
        self.locked_target_px = (px, py)   # pixel location at initial lock -- keeps
                                            # subsequent checks anchored to THIS object,
                                            # not just anything sharing the same name
        self.current_finger_targets = analysis.get("finger_joint_targets")
        self.correction_attempts = 0
        self.inconclusive_retries = 0
        self.move_to(target_base, then_verify=True)

    # ------------------------------------------------------------------
    # Step 2: execute a move, then schedule a verification check
    # ------------------------------------------------------------------
    def move_to(self, target_base, then_verify: bool):
        joint_angles = self.solve_ik(target_base)
        if joint_angles is None:
            self.get_logger().error("IK failed; aborting this pick attempt.")
            self.reset_to_idle()
            return

        self.publish_trajectory(joint_angles)
        self.state = PickState.MOVING

        if then_verify:
            wait_sec = MOVE_DURATION_SEC + SETTLE_MARGIN_SEC
            self._pending_timer = self.create_timer(
                wait_sec, self._on_settle_timer_fired
            )

    def _on_settle_timer_fired(self):
        self._cancel_pending_timer()
        self.request_alignment_check()

    # ------------------------------------------------------------------
    # Step 3: ask the VLM pipeline to check alignment
    # ------------------------------------------------------------------
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
            f"Requested VLM alignment check for '{self.current_object_name}' "
            f"(attempt {self.correction_attempts + 1}/{MAX_CORRECTION_ATTEMPTS})."
        )

        # Timeout guard in case the VLM process never replies.
        self._pending_timer = self.create_timer(
            VERIFY_TIMEOUT_SEC, self._on_verify_timeout
        )

    def _on_verify_timeout(self):
        self._cancel_pending_timer()
        if self.state == PickState.AWAITING_ALIGNMENT_CHECK:
            self.get_logger().warn(
                "No alignment check result received in time; giving up on "
                "this correction cycle."
            )
            self.reset_to_idle()

    # ------------------------------------------------------------------
    # Step 4: consume the VLM's alignment result and correct if needed
    # ------------------------------------------------------------------
    def on_alignment_result(self, msg: String):
        if self.state != PickState.AWAITING_ALIGNMENT_CHECK:
            return  # stale / unrelated result

        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if result.get("object_name") != self.current_object_name:
            return  # result for a different target; ignore

        self._cancel_pending_timer()

        gripper_visible = bool(result.get("gripper_visible", True))
        target_visible = bool(result.get("target_visible", True))

        if not gripper_visible or not target_visible:
            self.inconclusive_retries += 1
            if self.inconclusive_retries >= MAX_INCONCLUSIVE_RETRIES:
                self.get_logger().warn(
                    f"Gave up on alignment check for '{self.current_object_name}': "
                    f"gripper/target not visible after {self.inconclusive_retries} "
                    f"attempts (gripper_visible={gripper_visible}, "
                    f"target_visible={target_visible}). Check camera pose/FOV."
                )
                self.reset_to_idle()
                return

            # The VLM couldn't locate one of the two things it needs to
            # compare -- don't apply a correction based on a zero offset,
            # that would be a false "aligned". Just wait a moment and ask
            # again (doesn't count against MAX_CORRECTION_ATTEMPTS, since
            # this isn't a failed alignment, it's an inconclusive check).
            self.get_logger().warn(
                f"Alignment check inconclusive for '{self.current_object_name}' "
                f"(gripper_visible={gripper_visible}, target_visible={target_visible}). "
                f"Retrying check ({self.inconclusive_retries}/{MAX_INCONCLUSIVE_RETRIES}). "
                f"Notes: {result.get('notes', '')}"
            )
            self._pending_timer = self.create_timer(
                SETTLE_MARGIN_SEC, self._on_settle_timer_fired
            )
            self.state = PickState.MOVING  # reuse settle path to re-trigger check
            return

        aligned = bool(result.get("aligned", False))
        dx_px = float(result.get("offset_x_px", 0.0))
        dy_px = float(result.get("offset_y_px", 0.0))
        offset_mag_px = math.hypot(dx_px, dy_px)

        if aligned or offset_mag_px <= ALIGNMENT_TOLERANCE_PX:
            self.get_logger().info(
                f"Gripper aligned with '{self.current_object_name}' "
                f"(offset {offset_mag_px:.1f}px). Closing hand to grasp."
            )
            self.execute_grasp()
            self._mark_recently_grasped(*self.locked_target_px)
            self.reset_to_idle()
            return

        self.correction_attempts += 1
        if self.correction_attempts >= MAX_CORRECTION_ATTEMPTS:
            self.get_logger().warn(
                f"Max correction attempts reached for "
                f"'{self.current_object_name}' (last offset "
                f"{offset_mag_px:.1f}px). Stopping without full alignment."
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
            f"Misaligned by {offset_mag_px:.1f}px "
            f"(dx={dx_px:.1f}, dy={dy_px:.1f}). Correcting -> "
            f"({corrected_target[0]:.3f}, {corrected_target[1]:.3f}, "
            f"{corrected_target[2]:.3f})."
        )
        self.move_to(corrected_target, then_verify=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def pixel_offset_to_base_correction(self, dx_px, dy_px):
        """Convert a pixel offset in the overhead image into a small
        (x, y, z) nudge in the base_link frame, using the same
        ground-plane scale as the initial pixel_to_world projection."""
        camera_height_above_apples = CAMERA_WORLD_Z - APPLE_WORLD_Z
        m_per_px = camera_height_above_apples / FOCAL_LENGTH_PX

        # image dy -> world x, image dx -> world y (matches pixel_to_world)
        d_world_x = V_SIGN * (dy_px * m_per_px) * CORRECTION_GAIN
        d_world_y = U_SIGN * (dx_px * m_per_px) * CORRECTION_GAIN

        # World and base_link are assumed axis-aligned in x/y for this
        # correction step (base_footprint -> base_link is a small, mostly
        # translational offset). If your TF has significant rotation
        # between them, rotate this vector through that transform instead
        # of applying it directly.
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
                BASE_FRAME, WORLD_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None

    def solve_ik(self, target_base):
        try:
            ik_solution = self.chain.inverse_kinematics(
                target_position=np.array(target_base),
                target_orientation=np.array(TARGET_APPROACH_DIRECTION),
                orientation_mode=ORIENTATION_MODE,
            )
        except Exception as e:
            self.get_logger().error(f"IK solve failed: {e}")
            return None

        arm_angles = [ik_solution[i] for i in range(2, 8)]
        self.get_logger().info(f"IK solution (arm joints): {arm_angles}")
        return arm_angles

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

    def execute_grasp(self):
        """Close the hand using the finger_joint_targets captured at detection
        time. If none were captured (shouldn't normally happen), skip closing
        rather than commanding an undefined pose."""
        if not self.current_finger_targets:
            self.get_logger().warn(
                "No finger_joint_targets available; skipping hand close."
            )
            return

        positions = []
        for name in HAND_JOINTS:
            normalized = float(self.current_finger_targets.get(name, 0.0))
            normalized = max(0.0, min(1.0, normalized))
            lo, hi = HAND_JOINT_LIMITS_RAD[name]
            positions.append(lo + normalized * (hi - lo))

        traj = JointTrajectory()
        traj.joint_names = HAND_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(HAND_CLOSE_DURATION_SEC)
        point.time_from_start.nanosec = int((HAND_CLOSE_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)
        self.get_logger().info(f"Published hand-close trajectory: {positions}")

    def _mark_recently_grasped(self, px, py):
        now = self.get_clock().now().nanoseconds / 1e9
        self.recently_grasped.append((px, py, now))

    def _is_recently_grasped(self, px, py):
        now = self.get_clock().now().nanoseconds / 1e9
        # purge stale entries while we're at it
        self.recently_grasped = [
            (gx, gy, t) for (gx, gy, t) in self.recently_grasped
            if now - t < GRASP_COOLDOWN_SEC
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
