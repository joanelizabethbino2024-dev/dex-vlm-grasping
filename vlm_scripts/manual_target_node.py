#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from ikpy.chain import Chain
from rclpy.parameter import Parameter

ARM_TRAJECTORY_TOPIC = "/joint_trajectory_controller/joint_trajectory"
HAND_TRAJECTORY_TOPIC = "/dexhand_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanded_robot.urdf")
BASE_FRAME = "base_link"

ARM_JOINTS = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
HAND_JOINTS = ["R_Thumb_Pitch","R_Thumb_Roll","R_Thumb_Yaw","R_Thumb_Flexor","R_Thumb_DIP",
    "R_Index_Pitch","R_Index_Yaw","R_Index_Flexor","R_Index_DIP",
    "R_Middle_Pitch","R_Middle_Yaw","R_Middle_Flexor","R_Middle_DIP",
    "R_Ring_Pitch","R_Ring_Yaw","R_Ring_Flexor","R_Ring_DIP",
    "R_Pinky_Pitch","R_Pinky_Yaw","R_Pinky_Flexor","R_Pinky_DIP"]
# Real per-joint limits from expanded_robot.urdf (confirmed via URDF <limit> tags,
# 2026-09-08) -- Yaw/Roll joints have symmetric +/- ranges, NOT 0..90 deg like the
# curl joints, so the old uniform (0.0, pi/2) placeholder pushed every Yaw/Roll
# joint to roughly double its real limit when asked for "neutral" (0.5 normalized).
HAND_JOINT_LIMITS_RAD = {
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

TARGET_APPROACH_DIRECTION = [0, 0, 1]
ORIENTATION_MODE = "Z"

# The old model assumed the wrist-to-fingertip offset was a pure +Z vector
# of length HAND_LENGTH_OFFSET_M. That's wrong: orientation_mode="Z" only
# constrains the end effector's Z axis, leaving roll free, so ikpy is free
# to pick an arbitrary roll -- and it does, tilting the hand well off
# vertical (confirmed live: the two fingertips ended up ~8cm apart in Z for
# one IK solution). Measured directly via forward-kinematics + TF for the
# apple_05 target (base_link frame, wrist_3_link-ft_frame -> fingertip
# midpoint): (-0.005, -0.074, -0.114). Using this measured vector instead of
# a guessed pure-Z offset should track the actual hand geometry much more
# closely for targets that produce a similar arm/roll configuration to this
# one (e.g. other apples in the same cluster) -- it will NOT generalize to
# targets that make ikpy pick a very different roll; re-measure if so.
HAND_OFFSET_VECTOR_M = (-0.005, -0.074, -0.114)
HAND_LENGTH_OFFSET_M = 0.10  # kept only for DESCEND_OFFSET_M's fractional scaling below
# Bumped from 0.02: the descend target wasn't actually reaching the apple --
# confirmed live, contact was never detected during descent ("No contact
# detected... assuming target reached without obstruction"), meaning the
# fingers closed near the apple rather than overlapping it, which wasn't
# enough to hold on through the lift. Descending further increases overlap
# (contact detection still cuts it short if it hits the table/apple first).
DESCEND_OFFSET_M = 0.05
DESCEND_DURATION_SEC = 2.0
LIFT_DURATION_SEC = 7.0  # slower than a normal move (MOVE_DURATION_SEC) so a
                          # marginal grip isn't shaken loose by the lift itself
# Raises the wrist by rotating the upper arm back (negative = up, based on
# live data: shoulder_lift went 0.7142 (pre-curl height) -> 0.7277 (descended
# 5cm), i.e. INCREASING lowers the wrist). This is deliberately a plain joint
# delta, not an IK solve -- see publish_lift().
LIFT_SHOULDER_LIFT_DELTA_RAD = -0.3

PRE_CURL_AMOUNT = 0.5
PRE_CURL_DURATION_SEC = 1.5

# User's idea (2026-09-10): the apple keeps slipping out during the lift
# because contact only happens on a thin sliver of its surface -- a quick
# partial thumb curl BEFORE the full squeeze nudges the ball deeper into the
# palm/finger cage first (more wrap area committed before the fingers lock
# down), rather than asking the full close to both seat AND grip in one move.
THUMB_NUDGE_AMOUNT = 0.35
THUMB_NUDGE_DURATION_SEC = 1.5
NUDGE_SETTLE_SEC = 0.5

MOVE_DURATION_SEC = 4.0
HAND_CLOSE_DURATION_SEC = 4.0  # slower than before (was 2.0) -- a fast close can
                                # sweep/knock the object away before the fingers
                                # actually settle around it
DIP_CURL_CAP = 0.5  # cap DIP (fingertip joint) curl at half of Flexor's amount --
                     # curling the fingertip all the way can hook it PAST the
                     # object's surface instead of settling into a hook around it
LIFT_SETTLE_SEC = 1.0   # let fingers finish closing before lifting

# No real force/torque sensor is wired up yet, so we detect contact
# indirectly: after commanding a descend move, check whether the arm's
# ACTUAL joint positions match what we commanded. If a joint is physically
# blocked (e.g. the thumb hit the ground), it can't reach the commanded
# angle -- a real, measurable difference, not a guess. This is a proxy for
# contact, not a true force sensor -- consider it a placeholder for real
# force/torque sensing later.
CONTACT_CHECK_DELAY_SEC = 1.0   # how often to poll (sim seconds)
CONTACT_POSITION_TOLERANCE_RAD = 0.02   # error above this = "blocked"
# A freshly-published trajectory takes MOVE_DURATION_SEC of SIM time to
# interpolate to its target -- checking position error before that has
# elapsed will always show a "large" error simply because the move isn't
# finished yet, not because of contact. Confirmed live: checks at ~1
# sim-second into a 4-second move reported errors from 0.04 up to 2.46 rad
# on repeated identical runs, wildly inconsistent -- yet the FINAL settled
# position (measured after the move had time to complete) was consistently
# within ~0.05 rad of commanded. Don't trust the error signal as "contact"
# until most of the move should already be done.
CONTACT_CHECK_MIN_SEC = MOVE_DURATION_SEC * 0.7
CONTACT_CHECK_MAX_SEC = MOVE_DURATION_SEC + 1.0   # give up waiting for contact and
                                                    # close the hand anyway -- without
                                                    # this cap, a target the arm can
                                                    # reach without obstruction (no
                                                    # contact ever detected) polls
                                                    # forever and the node never
                                                    # finishes. This is compared against
                                                    # SIM time (see _descend_start_time),
                                                    # not wall-clock -- the descend
                                                    # trajectory itself is given
                                                    # MOVE_DURATION_SEC of SIM time to
                                                    # execute (publish_arm_trajectory
                                                    # always uses MOVE_DURATION_SEC), so
                                                    # a wall-clock cap would fire long
                                                    # before the arm physically finished
                                                    # moving whenever the sim's real-time
                                                    # factor is well under 1.0 (confirmed
                                                    # live at RTF~0.4: a 4s wall-clock cap
                                                    # was only ~1.6 sim-seconds, closing
                                                    # the hand while the arm was still
                                                    # mid-trajectory).


class ManualTargetNode(Node):
    def __init__(self, target_xyz, frame, close_hand, hand_delay, external_grasp=False, skip_close=False):
        super().__init__("manual_target_node", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.target_xyz = target_xyz
        self.frame = frame
        self.close_hand = close_hand
        self.hand_delay = hand_delay
        self.external_grasp = external_grasp
        self.skip_close = skip_close
        self.object_target = None
        self.pre_descend_target = None
        self.latest_joint_positions = {}
        self._pending_commanded_arm_angles = None
        self._descend_start_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base_link"])
        self.active_mask = [False]*len(self.chain.links)
        for i in range(2,8):
            if i < len(self.active_mask):
                self.active_mask[i] = True
        self.chain.active_links_mask = self.active_mask

        self.traj_pub = self.create_publisher(JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        self.hand_traj_pub = self.create_publisher(JointTrajectory, HAND_TRAJECTORY_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.joint_state_callback, 10)

        self.create_timer(1.0, self.run_once)
        self._done = False

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = position

    def _call_once_after(self, delay_sec, callback):
        """rclpy's create_timer() is periodic and keeps firing forever unless
        cancelled -- several stages here (previously) scheduled their next
        stage with a bare create_timer() and never stored/cancelled it, so
        the stage re-fired repeatedly (confirmed live: the whole
        descend->contact-check->grasp sequence ran twice on one target).
        This wraps a timer so it always cancels itself after firing once."""
        box = {}
        def _fire():
            timer = box.get("timer")
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
            callback()
        box["timer"] = self.create_timer(delay_sec, _fire)

    def run_once(self):
        if self._done: return
        self._done = True
        if self.frame == "base_link":
            target_base = self.target_xyz
        else:
            target_base = self.transform_to_base(self.target_xyz, self.frame)
            if target_base is None:
                self.get_logger().error(f"Could not transform from '{self.frame}' to '{BASE_FRAME}'.")
                rclpy.shutdown()
                return

        self.object_target = target_base

        adjusted_target = (
            target_base[0] - HAND_OFFSET_VECTOR_M[0],
            target_base[1] - HAND_OFFSET_VECTOR_M[1],
            target_base[2] - HAND_OFFSET_VECTOR_M[2],
        )
        self.pre_descend_target = adjusted_target

        self.get_logger().info(f"Object target: ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})")
        self.get_logger().info(f"IK target (hand-length offset): ({adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f})")

        joint_angles = self.solve_ik(adjusted_target)
        if joint_angles is None:
            self.get_logger().error("IK failed for this target -- likely unreachable.")
            rclpy.shutdown()
            return
        self.publish_arm_trajectory(joint_angles)
        if self.close_hand is not None:
            self.get_logger().info("Stage 1/4 done: initial approach. Will pre-curl fingers next...")
            self._call_once_after(MOVE_DURATION_SEC + self.hand_delay, self.publish_pre_curl)
        else:
            self._call_once_after(MOVE_DURATION_SEC + self.hand_delay, self.finish)

    def transform_to_base(self, xyz, source_frame):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = xyz
            transform = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None

    def solve_ik(self, target_base, seed_arm_angles=None):
        seed = [0.0] * len(self.chain.links)
        if seed_arm_angles is None:
            seed_arm_angles = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        for i, val in enumerate(seed_arm_angles):
            seed[2 + i] = val

        try:
            ik_solution = self.chain.inverse_kinematics(
                target_position=np.array(target_base),
                target_orientation=np.array(TARGET_APPROACH_DIRECTION),
                orientation_mode=ORIENTATION_MODE,
                initial_position=seed,
            )
        except Exception as e:
            self.get_logger().error(f"IK solve failed: {e}")
            return None
        return [ik_solution[i] for i in range(2, 8)]

    def publish_arm_trajectory(self, joint_angles, duration_sec=MOVE_DURATION_SEC):
        self._pending_commanded_arm_angles = joint_angles
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj.points.append(point)
        self.traj_pub.publish(traj)
        self.get_logger().info(f"Published arm trajectory: {joint_angles} (duration={duration_sec}s)")

    def _build_hand_positions(self, four_finger_amount, thumb_amount):
        # FIXED_RAW_OVERRIDES (raw radians, NOT the 0-1 "amount" convention below):
        # solved/confirmed via live testing (2026-09-10) so the fingers converge into
        # a self-centering cage and the thumb reaches as close to opposing them as its
        # kinematics allow, instead of every Yaw/Roll sitting at neutral (which was the
        # actual root cause of the apple getting shoved out sideways rather than
        # squeezed in place -- confirmed live: with everything neutral, the four
        # fingers close in parallel and the thumb doesn't face them at all).
        # R_Middle_Yaw is left out (stays neutral) -- it's the central reference
        # finger; Index/Ring/Pinky converge toward it.
        # R_Thumb_Yaw/Roll/Pitch are a fixed numerically-solved "reach toward the
        # object" pose (forward-kinematics fit against the apple's measured
        # position); only Thumb_Flexor/DIP stay dynamically driven by thumb_amount
        # for the actual squeeze, same as the other fingers.
        FIXED_RAW_OVERRIDES = {
            "R_Index_Yaw": 0.25, "R_Ring_Yaw": 0.25, "R_Pinky_Yaw": 0.25,
            "R_Thumb_Yaw": -0.523599, "R_Thumb_Roll": -0.125457, "R_Thumb_Pitch": 0.786401,
        }
        positions = []
        for name in HAND_JOINTS:
            lo, hi = HAND_JOINT_LIMITS_RAD[name]
            if name in FIXED_RAW_OVERRIDES:
                positions.append(FIXED_RAW_OVERRIDES[name])
                continue
            if "Yaw" in name or "Roll" in name:
                normalized = 0.5
            elif name.startswith("R_Thumb"):
                normalized = thumb_amount
            else:
                normalized = four_finger_amount
            if "DIP" in name:
                normalized = min(normalized, DIP_CURL_CAP)
            positions.append(hi - normalized * (hi - lo))
        return positions

    def publish_pre_curl(self):
        positions = self._build_hand_positions(four_finger_amount=PRE_CURL_AMOUNT, thumb_amount=0.0)
        traj = JointTrajectory()
        traj.joint_names = HAND_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(PRE_CURL_DURATION_SEC)
        point.time_from_start.nanosec = int((PRE_CURL_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)
        self.get_logger().info("Stage 2/4 done: pre-curled four fingers, thumb extended.")
        self._call_once_after(PRE_CURL_DURATION_SEC + 1.0, self.publish_descend)

    def publish_descend(self):
        # Move closer than the initial approach by DESCEND_OFFSET_M, but keep
        # applying (a fraction of) HAND_OFFSET_VECTOR_M like run_once() does --
        # otherwise this targets the WRIST at (nearly) the apple's own
        # position, and since the fingertips sit ~10-14cm beyond the wrist,
        # they overshoot past the apple by about a hand-length. (Confirmed
        # live: fingertips landed ~13cm past the apple with this missing.)
        descend_fraction = (HAND_LENGTH_OFFSET_M - DESCEND_OFFSET_M) / HAND_LENGTH_OFFSET_M
        descend_target = (
            self.object_target[0] - HAND_OFFSET_VECTOR_M[0] * descend_fraction,
            self.object_target[1] - HAND_OFFSET_VECTOR_M[1] * descend_fraction,
            self.object_target[2] - HAND_OFFSET_VECTOR_M[2] * descend_fraction,
        )
        self.get_logger().info(f"Descending to: ({descend_target[0]:.3f}, {descend_target[1]:.3f}, {descend_target[2]:.3f})")
        joint_angles = self.solve_ik(descend_target)
        if joint_angles is None:
            self.get_logger().error("IK failed for descend step; closing hand from current position instead.")
            self.publish_hand_close()
            return
        self.publish_arm_trajectory(joint_angles)
        self.get_logger().info("Stage 3/4: descending with thumb extended, checking for contact...")
        self._descend_start_time = self.get_clock().now()
        # Instead of blindly waiting the full move duration, check EARLY and
        # REPEATEDLY whether the arm has been physically stopped by contact.
        self._contact_check_timer = self.create_timer(CONTACT_CHECK_DELAY_SEC, self.check_for_contact)

    def check_for_contact(self):
        """Compare where we COMMANDED the arm to go against where it
        ACTUALLY is. A real mismatch means something is physically blocking
        it -- almost certainly contact -- so stop descending further."""
        if self._pending_commanded_arm_angles is None:
            return

        elapsed_sim_sec = (self.get_clock().now() - self._descend_start_time).nanoseconds / 1e9
        if elapsed_sim_sec > CONTACT_CHECK_MAX_SEC:
            self.get_logger().info(
                f"No contact detected within {CONTACT_CHECK_MAX_SEC}s -- assuming the "
                "descend target was reached without obstruction."
            )
            self._contact_check_timer.cancel()
            self._reached_descend_position()
            return

        max_error = 0.0
        for name, commanded in zip(ARM_JOINTS, self._pending_commanded_arm_angles):
            actual = self.latest_joint_positions.get(name)
            if actual is None:
                return  # don't have fresh data yet, try again next tick
            max_error = max(max_error, abs(actual - commanded))

        if elapsed_sim_sec < CONTACT_CHECK_MIN_SEC:
            # Too early to trust this reading -- the trajectory is still
            # interpolating toward its target, so a "large" error here is
            # normal in-transit lag, not contact. Just keep polling.
            self.get_logger().info(
                f"Move still in progress ({elapsed_sim_sec:.1f}/{CONTACT_CHECK_MIN_SEC:.1f}s, "
                f"error {max_error:.4f} rad) -- too early to judge contact."
            )
            return

        if max_error > CONTACT_POSITION_TOLERANCE_RAD:
            self.get_logger().info(
                f"Contact detected (position error {max_error:.4f} rad > "
                f"{CONTACT_POSITION_TOLERANCE_RAD} tolerance) -- stopping descent."
            )
            self._contact_check_timer.cancel()
            self._reached_descend_position()
            return

        # No contact yet -- keep checking, up to the normal move duration as a cap
        self.get_logger().info(f"No contact yet (position error {max_error:.4f} rad). Still checking...")

    def _reached_descend_position(self):
        if self.external_grasp:
            self.get_logger().info(
                "external_grasp mode: leaving the hand open and parked here for an "
                "external grasp controller (e.g. adaptive_grasp_controller.py) to take "
                "over. Shutting this node down; the arm/hand will hold this pose."
            )
            self.finish()
        elif self.skip_close:
            self.get_logger().info(
                "skip_close mode: going straight to lift with whatever grip the "
                "pre-curl pose already has -- no thumb nudge, no full close -- to "
                "isolate whether the squeeze motions themselves knock the object loose."
            )
            self.publish_lift()
        else:
            self.publish_thumb_nudge()

    def publish_thumb_nudge(self):
        """Stage 3.5/4: curl just the thumb in partway (fingers stay at their
        pre-curl amount) to nudge the object deeper into the palm/finger cage
        before the full squeeze locks everything down."""
        positions = self._build_hand_positions(
            four_finger_amount=PRE_CURL_AMOUNT, thumb_amount=THUMB_NUDGE_AMOUNT
        )
        traj = JointTrajectory()
        traj.joint_names = HAND_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(THUMB_NUDGE_DURATION_SEC)
        point.time_from_start.nanosec = int((THUMB_NUDGE_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)
        self.get_logger().info("Stage 3.5/4: nudging object inward with thumb before full close...")
        self._call_once_after(THUMB_NUDGE_DURATION_SEC + NUDGE_SETTLE_SEC, self.publish_hand_close)

    def publish_hand_close(self):
        positions = self._build_hand_positions(four_finger_amount=self.close_hand, thumb_amount=self.close_hand)
        traj = JointTrajectory()
        traj.joint_names = HAND_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(HAND_CLOSE_DURATION_SEC)
        point.time_from_start.nanosec = int((HAND_CLOSE_DURATION_SEC % 1) * 1e9)
        traj.points.append(point)
        self.hand_traj_pub.publish(traj)
        self.get_logger().info("Stage 4/4 done: full grip (thumb now closing too).")
        self._call_once_after(HAND_CLOSE_DURATION_SEC + LIFT_SETTLE_SEC, self.publish_lift)

    def publish_lift(self):
        """Stage 5/5: raise the arm off the ground. If the grasp actually
        holds, the object comes up with the hand -- this is the real pick-up
        test, not just contact. Check the object's world pose before/after
        this (e.g. via `ign topic -e -t /world/<world>/pose/info`) to confirm
        it rose along with the hand rather than staying on the table/getting
        knocked away.

        Deliberately NOT an IK solve: re-solving IK (even seeded with the
        current angles, even with wrist_3 pinned) is still liable to shift
        every joint slightly, and any shift can disturb a marginal grip.
        Instead this only changes ONE joint (shoulder_lift_joint) by a fixed
        delta from whatever the arm is CURRENTLY holding -- every other arm
        joint and every hand joint stays byte-identical to the grip pose."""
        if self._pending_commanded_arm_angles is None:
            self.get_logger().warn("No commanded arm angles recorded; skipping lift.")
            self._call_once_after(1.0, self.finish)
            return
        joint_angles = list(self._pending_commanded_arm_angles)
        joint_angles[1] += LIFT_SHOULDER_LIFT_DELTA_RAD
        self.publish_arm_trajectory(joint_angles, duration_sec=LIFT_DURATION_SEC)
        self.get_logger().info(
            f"Stage 5/5: lifting via shoulder_lift_joint only (delta={LIFT_SHOULDER_LIFT_DELTA_RAD} rad), "
            "every other joint unchanged from the grip pose..."
        )
        self._call_once_after(LIFT_DURATION_SEC + 1.0, self.finish)

    def finish(self):
        self.get_logger().info("Done. Shutting down.")
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--frame", type=str, default="base_link")
    parser.add_argument("--close-hand", type=float, default=None)
    parser.add_argument("--hand-delay", type=float, default=5.0)
    parser.add_argument(
        "--external-grasp", action="store_true",
        help="Position and descend as usual, but leave the hand open and let an "
             "external grasp controller (e.g. adaptive_grasp_controller.py) close "
             "it, instead of this node's own fixed-position close.",
    )
    parser.add_argument(
        "--skip-close", action="store_true",
        help="Descend and detect contact as usual, but skip the thumb-nudge and "
             "full-close stages -- go straight to lift with whatever grip the "
             "pre-curl pose already has. Diagnostic: isolates whether the squeeze "
             "motions themselves are what knock the object loose.",
    )
    args = parser.parse_args()
    rclpy.init()
    node = ManualTargetNode(
        (args.x, args.y, args.z), args.frame, args.close_hand, args.hand_delay,
        external_grasp=args.external_grasp, skip_close=args.skip_close,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
