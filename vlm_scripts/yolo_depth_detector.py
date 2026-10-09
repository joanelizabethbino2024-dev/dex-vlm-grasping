#!/usr/bin/env python3
"""
YOLO + Depth Camera Detector (gripper-aware)
-----------------------------------------------
Does NOT trust YOLO's class labels (confirmed unreliable on this simulated
scene -- it calls the apple/arm "airplane"/"kite"). Instead:

  1. Accept every YOLO detection above a low confidence bar, regardless of
     its (untrustworthy) label.
  2. Compute the GRIPPER's real position directly from the robot's own
     joint angles via forward kinematics -- deterministic, no vision
     involved, can't be mislabeled.
  3. For each YOLO detection, compute its real 3D position via depth +
     camera intrinsics + TF (the pipeline we just got working).
  4. Drop any detection whose position is close to the gripper's own
     position -- that's almost certainly the arm/hand itself, not a
     pickable object.
  5. Whatever's left are genuine candidate objects. Publish the one
     nearest the image center.
"""
import json
import math
import os
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration

from ultralytics import YOLO
from ikpy.chain import Chain

RGB_TOPIC = "/overhead_camera"
DEPTH_TOPIC = "/overhead_rgbd/depth_image"
CAMERA_INFO_TOPIC = "/overhead_rgbd/camera_info"
RESULT_TOPIC = "/overhead_camera/fragility_analysis"
JOINT_STATES_TOPIC = "/joint_states"

BASE_FRAME = "base_link"
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanded_robot.urdf")
YOLO_WEIGHTS_PATH = os.path.expanduser("~/yolov8n.pt")

# Overhead camera is a static, un-articulated model in the world SDF (not
# part of the robot's URDF/TF tree), so its world pose isn't observable via
# TF -- this is a real extrinsic calibration value (matches apple_world.world's
# <model name="overhead_depth_camera"><pose>), not a guessed constant.
CAMERA_WORLD_X = 1.125
CAMERA_WORLD_Y = 0.0
CAMERA_WORLD_Z = 2.0
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
# Fallback intrinsics, only used if a CameraInfo message hasn't arrived yet.
# Once camera_info_callback fires, the real per-axis fx/fy/cx/cy from the
# sensor's own calibration are used instead (see pixel_and_depth_to_base).
HORIZONTAL_FOV_RAD = 1.396
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)
WORLD_FRAME = "base_footprint"

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

DETECT_INTERVAL_SEC = 0.5
CONFIDENCE_THRESHOLD = 0.3          # accept ANY object above this, regardless of label
GRIPPER_EXCLUSION_RADIUS_M = 0.20   # candidates within this of the gripper END-EFFECTOR POINT
                                     # are excluded. This alone is NOT enough: the arm's other
                                     # links (forearm, upper arm) are also visible from directly
                                     # overhead and are NOT near this one point, so a YOLO box
                                     # drawn on the arm itself sails straight through this check.
                                     # See the real-world size filter below for the actual guard
                                     # against that (confirmed live 2026-09-15: an unfiltered
                                     # "apple" detection back-computed to ~1.2m wide -- the arm,
                                     # not the apple).

# apple_05's real collision radius (apple_gripper_sim/models/apple_05/model.sdf) is 0.04m,
# i.e. an 0.08m diameter. Reject any candidate whose real-world width -- computed from its
# OWN measured depth, not guessed -- falls outside a generous band around that. This is what
# actually rejects the robot's own arm/gripper (meters-wide) and sensor-noise slivers,
# independent of how far they happen to sit from the single gripper exclusion point above.
MIN_OBJECT_WIDTH_M = 0.02
MAX_OBJECT_WIDTH_M = 0.16

# Only used for the "no object visible" zeroed-targets message below -- real
# detections get a SIZE-ADAPTIVE pre-shape from finger_targets_for_size()
# instead (previously this one fixed dict was used for every object
# regardless of size).
DEFAULT_FINGER_TARGETS = {
    "R_Thumb_Pitch": 0.4, "R_Thumb_Roll": 0.5, "R_Thumb_Yaw": 0.5,
    "R_Thumb_Flexor": 0.4, "R_Thumb_DIP": 0.4,
    "R_Index_Pitch": 0.4, "R_Index_Yaw": 0.5, "R_Index_Flexor": 0.4, "R_Index_DIP": 0.4,
    "R_Middle_Pitch": 0.4, "R_Middle_Yaw": 0.5, "R_Middle_Flexor": 0.4, "R_Middle_DIP": 0.4,
    "R_Ring_Pitch": 0.2, "R_Ring_Yaw": 0.5, "R_Ring_Flexor": 0.2, "R_Ring_DIP": 0.2,
    "R_Pinky_Pitch": 0.2, "R_Pinky_Yaw": 0.5, "R_Pinky_Flexor": 0.2, "R_Pinky_DIP": 0.2,
}

# --- Size-adaptive finger pre-shape, built on a PROVEN base pose ---
# manual_target_node.py's live pick-and-lift testing (2026-09-10) found
# NEUTRAL Yaw for every finger let the fingers close in parallel with the
# thumb not facing them at all -- the object got shoved sideways instead of
# caged. Its confirmed fix was specific non-neutral Yaw/Roll values (here
# converted from the raw radians it uses into this file's normalized
# hi-value*(hi-lo) convention) so the fingers converge into a self-centering
# cage with the thumb opposing them. That opposition geometry is kept as the
# BASE below, not the final output -- Pitch (curl amount) and the additional
# Yaw SPREAD on top of this base still scale with the object's own measured
# real_size_m, so a differently-sized object later still gets a
# differently-shaped hand, not an identical pose every time.
#
# Flexor/DIP (actual curl) are left at 0.0 (open, hi=open/lo=closed
# convention -- see adaptive_grasp_controller.py's normalized_to_radians)
# here -- open_hand_preshape() forces those open regardless of what's in
# this dict, and adaptive_grasp_controller.py's force-feedback squeeze
# decides the real closing amount live, not this pre-shape.
#
# NOTE: the proven base was measured for ONE specific apple/approach-roll
# combination (manual_target_node.py's own comment: "will NOT generalize to
# targets that make ikpy pick a very different roll -- re-measure if so").
# The size-scaling on top is this project's own addition, not independently
# live-validated across a real object-size sweep yet.
YAW_BASE_OUTER = 0.142     # Index/Ring/Pinky Yaw baseline (proven opposition pose)
THUMB_YAW_BASE = 1.0
THUMB_ROLL_BASE = 0.680
MAX_YAW_SPREAD = 0.15       # additional normalized spread on top of the base at max graspable size
PITCH_SMALL_OBJECT = 0.45   # more curl for a small object -- reach in closer before squeeze
PITCH_LARGE_OBJECT = 0.20   # less curl for a large object -- stay open wider before squeeze


def finger_targets_for_size(real_size_m):
    lo, hi = MIN_OBJECT_WIDTH_M, MAX_OBJECT_WIDTH_M
    size_fraction = max(0.0, min(1.0, (real_size_m - lo) / (hi - lo))) if hi > lo else 0.5

    pitch = PITCH_SMALL_OBJECT + size_fraction * (PITCH_LARGE_OBJECT - PITCH_SMALL_OBJECT)
    spread = size_fraction * MAX_YAW_SPREAD

    return {
        "R_Thumb_Pitch": pitch, "R_Thumb_Roll": THUMB_ROLL_BASE, "R_Thumb_Yaw": THUMB_YAW_BASE,
        "R_Thumb_Flexor": 0.0, "R_Thumb_DIP": 0.0,
        "R_Index_Pitch": pitch, "R_Index_Yaw": YAW_BASE_OUTER + spread,
        "R_Index_Flexor": 0.0, "R_Index_DIP": 0.0,
        "R_Middle_Pitch": pitch, "R_Middle_Yaw": 0.5,
        "R_Middle_Flexor": 0.0, "R_Middle_DIP": 0.0,
        "R_Ring_Pitch": pitch, "R_Ring_Yaw": YAW_BASE_OUTER - spread,
        "R_Ring_Flexor": 0.0, "R_Ring_DIP": 0.0,
        "R_Pinky_Pitch": pitch, "R_Pinky_Yaw": YAW_BASE_OUTER - spread,
        "R_Pinky_Flexor": 0.0, "R_Pinky_DIP": 0.0,
    }


class YoloDepthDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_depth_detector_node")
        self.bridge = CvBridge()

        self.get_logger().info(f"Loading YOLOv8n from {YOLO_WEIGHTS_PATH}...")
        self.model = YOLO(YOLO_WEIGHTS_PATH)
        self.get_logger().info("YOLO model loaded.")

        self.get_logger().info("Loading IK chain for gripper forward kinematics...")
        self.chain = Chain.from_urdf_file(URDF_PATH, base_elements=["base_link"])
        self.get_logger().info(f"Chain loaded with {len(self.chain.links)} links.")

        self.latest_depth = None
        self.depth_frame_id = None
        self.camera_intrinsics = None
        self.latest_joint_positions = {}
        self.last_detect_time = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.joint_state_callback, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)

        self.get_logger().info(f"Subscribed to {RGB_TOPIC}, {DEPTH_TOPIC}, {CAMERA_INFO_TOPIC}, {JOINT_STATES_TOPIC}")
        self.get_logger().info(f"Publishing detections to {RESULT_TOPIC}")

    def camera_info_callback(self, msg):
        k = msg.k
        self.camera_intrinsics = (k[0], k[4], k[2], k[5])

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = position

    def compute_gripper_position(self):
        """Real gripper position via forward kinematics from actual joint
        angles -- deterministic, not vision, can't be mislabeled."""
        for name in ARM_JOINTS:
            if name not in self.latest_joint_positions:
                return None

        joint_array = [0.0] * len(self.chain.links)
        for i, name in enumerate(ARM_JOINTS):
            joint_array[2 + i] = self.latest_joint_positions[name]

        fk_transform = self.chain.forward_kinematics(joint_array)
        position = fk_transform[:3, 3]
        return (float(position[0]), float(position[1]), float(position[2]))

    def rgb_callback(self, msg):
        now = time.time()
        if now - self.last_detect_time < DETECT_INTERVAL_SEC:
            return
        self.last_detect_time = now

        if self.latest_depth is None or self.camera_intrinsics is None:
            self.get_logger().warn("Waiting for depth + camera_info...", throttle_duration_sec=5)
            return

        gripper_pos = self.compute_gripper_position()
        if gripper_pos is None:
            self.get_logger().warn("Waiting for joint states to compute gripper position...", throttle_duration_sec=5)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return

        results = self.model(cv_image, verbose=False)[0]

        candidates = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            point_base = self.pixel_and_depth_to_base(cx, cy)
            if point_base is None:
                continue

            dist_to_gripper = math.dist(point_base, gripper_pos)
            if dist_to_gripper <= GRIPPER_EXCLUSION_RADIUS_M:
                continue  # this is almost certainly the arm/gripper itself

            real_size_m = self.estimate_real_size_m(cx, cy, max(x2 - x1, y2 - y1))
            if real_size_m is None or not (MIN_OBJECT_WIDTH_M <= real_size_m <= MAX_OBJECT_WIDTH_M):
                continue  # not apple-sized -- almost certainly the arm/gripper body or noise

            candidates.append({
                "cx": cx, "cy": cy, "width_px": x2 - x1,
                "confidence": conf, "point_base": point_base, "real_size_m": real_size_m,
                "label": self.model.names[int(box.cls[0])],  # kept for logging only, not trusted
            })

        if not candidates:
            self.publish_none_visible()
            return

        img_cx, img_cy = 320, 240
        best = min(candidates, key=lambda c: math.hypot(c["cx"] - img_cx, c["cy"] - img_cy))
        self.publish_detection(best)

    def pixel_and_depth_to_base(self, px, py):
        """x/y via the depth camera's OWN calibrated intrinsics (fx, fy, cx,
        cy from its CameraInfo message) rather than an assumed FOV/centered
        principal point -- more precise, and self-correcting if the sensor's
        real FOV or optical center ever drifts from the nominal values. z
        comes from the depth camera's raw scalar distance reading, which is
        a REAL measurement, not an assumed fixed apple height."""
        h, w = self.latest_depth.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        depth = float(self.latest_depth[iy, ix])
        if not math.isfinite(depth) or depth <= 0.0 or depth > CAMERA_WORLD_Z:
            return None  # implausible reading -- discard rather than trust it

        if self.camera_intrinsics is not None:
            fx, fy, cx, cy = self.camera_intrinsics
        else:
            fx = fy = FOCAL_LENGTH_PX
            cx, cy = IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0

        dx_px = px - cx
        dy_px = py - cy
        # SIGN CONFIRMED (2026-09-16) via live ground truth: apple_05 spawned at
        # world (1.0, 0.0, 0.04); the overhead camera's own image showed it at
        # pixel (320, 266), not the (320, 216) a "+" sign here predicts. Only
        # "-" reproduces the real position (predicted world_x ~0.99 vs the
        # true 1.0). The camera's optical axes are rotated 90 deg from the
        # naive assumption, so both offsets need the flip.
        world_x = CAMERA_WORLD_X - (dy_px * depth / fy)
        world_y = CAMERA_WORLD_Y - (dx_px * depth / fx)
        world_z = CAMERA_WORLD_Z - depth   # REAL measured height, not assumed

        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = WORLD_FRAME
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = world_x, world_y, world_z
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, WORLD_FRAME, rclpy.time.Time(), timeout=Duration(seconds=2.0)
            )
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None

    def estimate_real_size_m(self, px, py, size_px):
        """Convert a bbox's pixel extent to a real-world size in meters using
        the SAME measured depth and calibrated focal length used for
        position -- no separate assumption needed. Lets us reject boxes that
        are the wrong physical size to be an apple (the robot's own arm,
        seen from overhead, is meters long; sensor noise slivers are
        sub-pixel) regardless of where they sit relative to the gripper."""
        h, w = self.latest_depth.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        depth = float(self.latest_depth[iy, ix])
        if not math.isfinite(depth) or depth <= 0.0 or depth > CAMERA_WORLD_Z:
            return None

        fx = self.camera_intrinsics[0] if self.camera_intrinsics is not None else FOCAL_LENGTH_PX
        return size_px * depth / fx

    def publish_none_visible(self):
        result = {
            "object_name": "none visible", "description": "", "fragility_score": 0,
            "finger_joint_targets": {k: 0.0 for k in DEFAULT_FINGER_TARGETS},
            "confidence": 0.0, "notes": "no non-gripper candidates found this frame",
            "bbox_center_x": 320, "bbox_center_y": 240, "bbox_width_px": 0,
        }
        msg = String()
        msg.data = json.dumps(result)
        self.result_pub.publish(msg)

    def publish_detection(self, c):
        pb = c["point_base"]
        finger_targets = finger_targets_for_size(c["real_size_m"])
        result = {
            "object_name": "apple",
            "description": f"candidate object (YOLO called it '{c['label']}', not trusted), conf {c['confidence']:.2f}",
            "fragility_score": 4,
            "finger_joint_targets": finger_targets,
            "confidence": c["confidence"],
            "notes": (
                f"3D position (base_link): ({pb[0]:.3f}, {pb[1]:.3f}, {pb[2]:.3f}), "
                f"measured real size: {c['real_size_m']:.3f}m"
            ),
            "bbox_center_x": int(c["cx"]),
            "bbox_center_y": int(c["cy"]),
            "bbox_width_px": int(c["width_px"]),
            "base_link_x": pb[0], "base_link_y": pb[1], "base_link_z": pb[2],
            "real_size_m": c["real_size_m"],
        }
        msg = String()
        msg.data = json.dumps(result)
        self.result_pub.publish(msg)
        print(f'>>> apple: ({pb[0]:.3f}, {pb[1]:.3f}, {pb[2]:.3f}), size={c["real_size_m"]:.3f}m', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
