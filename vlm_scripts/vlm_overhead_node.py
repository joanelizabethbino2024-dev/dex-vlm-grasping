#!/usr/bin/env python3
"""
VLM Overhead Camera Node
--------------------------
Same Ollama/qwen2.5vl integration as the existing gripper-camera fragility
node, but for the FIXED OVERHEAD camera, with one addition: it can answer
on-demand "is the gripper aligned with object X?" requests, which is what
the closed-loop IK node (vision_ik_overhead_closed_loop.py) needs after
each move.

Two independent jobs, same running node:

  1. Periodic grasp analysis (same as before): every QUERY_INTERVAL_SEC,
     grab the latest overhead frame, ask the VLM to find/describe the
     object and propose a grasp, publish to RESULT_TOPIC.

  2. On-demand alignment check: when a JSON request arrives on
     ALIGNMENT_REQUEST_TOPIC (published by the IK node right after a move
     settles), immediately grab the latest cached frame, ask the VLM to
     locate BOTH the gripper/end-effector and the named object in pixel
     space, compute the offset between them, and publish the result to
     ALIGNMENT_RESULT_TOPIC.

IMPORTANT - CONFIRM THIS: IMAGE_TOPIC below is a guess
(`/overhead_camera/image_raw`). Point it at whatever topic your overhead
camera sensor/plugin actually publishes raw images on -- the IK node's
comments imply a camera named `/overhead_camera`, but the exact image
topic wasn't in what you've shared so far.
"""
import base64
import json
import math
import os
import time

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

IMAGE_TOPIC = "/overhead_camera"                    # matches the bridged Ignition topic name
RESULT_TOPIC = "/overhead_camera/fragility_analysis"
ALIGNMENT_REQUEST_TOPIC = "/overhead_camera/alignment_check_request"
ALIGNMENT_RESULT_TOPIC = "/overhead_camera/alignment_check_result"

QUERY_INTERVAL_SEC = 2.0
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480

ALIGNMENT_TOLERANCE_PX = 8.0

# When a real depth-based detector (yolo_depth_detector.py) is running as the
# initial-detection source, this node's periodic pixel-only grasp analysis
# (job 1 above) would otherwise race it on RESULT_TOPIC and occasionally hand
# the IK node a less precise, assumed-height target. Set to disable job 1
# while keeping job 2 (on-demand alignment check, which this node is still
# needed for -- the depth detector doesn't do vision-language alignment).
PERIODIC_GRASP_ANALYSIS_ENABLED = os.environ.get("VLM_OVERHEAD_ALIGNMENT_ONLY", "0") != "1"

GRASP_PROMPT = """You are a robotic grasping assistant for a 5-finger dexterous robot hand
(thumb, index, middle, ring, pinky), each finger with a knuckle, middle, and tip joint.
An overhead camera looks straight down at a workspace containing a robot arm and one or
more small pickable objects (e.g. apples). Analyze the image and decide how to pick up
the target object safely.

IMPORTANT: The robot arm, gripper/hand, and any mounting hardware or mechanical
structures visible in the frame are NOT the target -- ignore them completely, even if
they are large or prominent in the image. Only report on the small, separate pickable
object(s) sitting in the workspace (e.g. apples on a surface). If multiple candidate
objects are visible, pick the one that is most clearly a graspable item (round fruit,
etc.) rather than robot hardware, and prefer the one nearest the image center if there
are several equally plausible candidates.

Respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

{
  "object_name": "best guess at what the object is",
  "description": "one short sentence describing size, shape, color, condition",
  "estimated_material": "e.g. glass, plastic, fruit skin, metal, ceramic, fabric",
  "fragility_score": 0-10 integer, where 0 is indestructible and 10 is extremely fragile,
  "surface_texture": "e.g. smooth, rough, bumpy, soft",
  "recommended_grip_force": "low, medium, or high",
  "grasp_method": "e.g. precision pinch (thumb+index), tripod grasp, power grasp, lateral pinch",
  "approach_vector": "e.g. top-down, side approach 30deg, front-on",
  "finger_joint_targets": {
    "R_Thumb_Pitch": 0.0-1.0, "R_Thumb_Roll": 0.0-1.0, "R_Thumb_Yaw": 0.0-1.0,
    "R_Thumb_Flexor": 0.0-1.0, "R_Thumb_DIP": 0.0-1.0,
    "R_Index_Pitch": 0.0-1.0, "R_Index_Yaw": 0.0-1.0,
    "R_Index_Flexor": 0.0-1.0, "R_Index_DIP": 0.0-1.0,
    "R_Middle_Pitch": 0.0-1.0, "R_Middle_Yaw": 0.0-1.0,
    "R_Middle_Flexor": 0.0-1.0, "R_Middle_DIP": 0.0-1.0,
    "R_Ring_Pitch": 0.0-1.0, "R_Ring_Yaw": 0.0-1.0,
    "R_Ring_Flexor": 0.0-1.0, "R_Ring_DIP": 0.0-1.0,
    "R_Pinky_Pitch": 0.0-1.0, "R_Pinky_Yaw": 0.0-1.0,
    "R_Pinky_Flexor": 0.0-1.0, "R_Pinky_DIP": 0.0-1.0
  },
  "confidence": 0.0-1.0 float representing how confident you are in this analysis,
  "notes": "one short sentence with any additional relevant observation",
  "bbox_center_x": 0-640 integer, horizontal pixel coordinate of the object's center (0=left edge, 640=right edge),
  "bbox_center_y": 0-480 integer, vertical pixel coordinate of the object's center (0=top edge, 480=bottom edge),
  "bbox_width_px": integer, approximate width in pixels of the object as it appears in the image
}

Guidance for finger_joint_targets: 0.0 = fully open/straight, 1.0 = fully curled/closed
(0.5 = neutral for Yaw/Roll joints). Use lighter Flexor/DIP values and a tripod grasp for
fragile/soft objects; fuller power grasp closure for rigid/heavy objects.

If no pickable object (other than the arm/gripper itself) is clearly visible, set
object_name to "none visible", fragility_score to 0, and all finger_joint_targets to 0.0
(0.5 for Yaw/Roll joints).

Respond with ONLY the JSON object, nothing else."""

ALIGNMENT_PROMPT_TEMPLATE = """You are helping a robot arm verify its position using a fixed
overhead camera looking straight down at a workspace. The image is {width}x{height} pixels
(0,0 = top-left).

Two things need to be located in THIS image:
  1. The robot's gripper / end-effector / hand (it may be partially visible entering the
     frame from above, or fully visible if it has reached down to the workspace). This
     means the FINGERS/HAND specifically -- not the arm links, wrist, or forearm leading
     up to it.
  2. The target object, described as: "{object_name}". This is a small pickable item
     (e.g. an apple), NOT any part of the robot arm or mounting hardware, even if such
     hardware is large or prominent in the frame.
{pixel_hint_line}

Respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

{{
  "gripper_visible": true or false,
  "gripper_center_x": 0-{width} integer pixel x of the gripper/hand center, or null if not visible,
  "gripper_center_y": 0-{height} integer pixel y of the gripper/hand center, or null if not visible,
  "target_visible": true or false,
  "target_center_x": 0-{width} integer pixel x of the target object's center, or null if not visible,
  "target_center_y": 0-{height} integer pixel y of the target object's center, or null if not visible,
  "confidence": 0.0-1.0 float,
  "notes": "one short sentence, e.g. why something wasn't visible"
}}

Be as accurate as possible with the pixel coordinates -- they are used to physically
correct the robot's position. If the gripper is not visible in the frame, set
gripper_visible to false and leave gripper_center_x/y as null; do not guess.

Respond with ONLY the JSON object, nothing else."""


class VLMOverheadNode(Node):
    def __init__(self):
        super().__init__("vlm_overhead_node")
        self.bridge = CvBridge()

        self.latest_cv_image = None
        self.last_periodic_query_time = 0.0

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.create_subscription(
            String, ALIGNMENT_REQUEST_TOPIC, self.on_alignment_request, 10
        )

        self.grasp_pub = self.create_publisher(String, RESULT_TOPIC, 10)
        self.alignment_pub = self.create_publisher(String, ALIGNMENT_RESULT_TOPIC, 10)

        self.get_logger().info(f"Subscribed to {IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing grasp analysis to {RESULT_TOPIC}")
        self.get_logger().info(
            f"Listening for alignment requests on {ALIGNMENT_REQUEST_TOPIC}"
        )
        self.get_logger().info(f"Using Ollama model: {MODEL_NAME}")
        if not PERIODIC_GRASP_ANALYSIS_ENABLED:
            self.get_logger().info(
                "VLM_OVERHEAD_ALIGNMENT_ONLY=1 -- periodic grasp analysis disabled, "
                "running alignment-check duty only."
            )

    # ------------------------------------------------------------------
    # Camera feed: cache latest frame, and (rate-limited) run grasp analysis
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        self.latest_cv_image = cv_image

        if not PERIODIC_GRASP_ANALYSIS_ENABLED:
            return

        now = time.time()
        if now - self.last_periodic_query_time < QUERY_INTERVAL_SEC:
            return
        self.last_periodic_query_time = now

        mean_val = cv_image.mean()
        if mean_val < 5.0:
            self.get_logger().warning(
                f"Incoming frame looks almost black (mean pixel={mean_val:.2f}). "
                "Check overhead camera pose/lighting in the world file."
            )
            return

        image_b64 = self._encode_image(cv_image)
        if image_b64 is None:
            return

        analysis = self.query_vlm(GRASP_PROMPT, image_b64)
        if analysis is not None:
            analysis = self._backfill_grasp_defaults(analysis)
            result_msg = String()
            result_msg.data = json.dumps(analysis)
            self.grasp_pub.publish(result_msg)
            self.get_logger().info(f"Grasp analysis: {analysis}")

    # ------------------------------------------------------------------
    # On-demand alignment check
    # ------------------------------------------------------------------
    def on_alignment_request(self, msg: String):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Alignment request was not valid JSON; ignoring.")
            return

        object_name = request.get("object_name")
        target_pixel_hint = request.get("target_pixel_hint")
        if not object_name:
            self.get_logger().warning("Alignment request missing object_name; ignoring.")
            return

        if self.latest_cv_image is None:
            self.get_logger().warning(
                "No overhead frame received yet; cannot run alignment check."
            )
            self._publish_alignment_result(
                object_name, gripper_visible=False, target_visible=False,
                offset_x_px=0.0, offset_y_px=0.0, aligned=False, confidence=0.0,
                notes="no camera frame available",
            )
            return

        image_b64 = self._encode_image(self.latest_cv_image)
        if image_b64 is None:
            return

        prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
            width=IMAGE_WIDTH_PX, height=IMAGE_HEIGHT_PX, object_name=object_name,
            pixel_hint_line=self._format_pixel_hint(target_pixel_hint),
        )
        result = self.query_vlm(prompt, image_b64)
        if result is None:
            self._publish_alignment_result(
                object_name, gripper_visible=False, target_visible=False,
                offset_x_px=0.0, offset_y_px=0.0, aligned=False, confidence=0.0,
                notes="VLM query failed",
            )
            return

        gripper_visible = bool(result.get("gripper_visible", False))
        target_visible = bool(result.get("target_visible", False))
        confidence = float(result.get("confidence", 0.0))
        notes = result.get("notes", "")

        if not gripper_visible or not target_visible:
            self._publish_alignment_result(
                object_name, gripper_visible=gripper_visible,
                target_visible=target_visible, offset_x_px=0.0, offset_y_px=0.0,
                aligned=False, confidence=confidence, notes=notes,
            )
            return

        gx = float(result.get("gripper_center_x", IMAGE_WIDTH_PX / 2))
        gy = float(result.get("gripper_center_y", IMAGE_HEIGHT_PX / 2))
        tx = float(result.get("target_center_x", IMAGE_WIDTH_PX / 2))
        ty = float(result.get("target_center_y", IMAGE_HEIGHT_PX / 2))

        # Remaining pixel error the gripper still needs to move to reach the
        # target -- same sign convention as the IK node's pixel_to_world.
        offset_x_px = tx - gx
        offset_y_px = ty - gy
        offset_mag_px = math.hypot(offset_x_px, offset_y_px)
        aligned = offset_mag_px <= ALIGNMENT_TOLERANCE_PX

        self._publish_alignment_result(
            object_name, gripper_visible=True, target_visible=True,
            offset_x_px=offset_x_px, offset_y_px=offset_y_px, aligned=aligned,
            confidence=confidence, notes=notes,
            gripper_px=(gx, gy), target_px=(tx, ty),
        )

    def _format_pixel_hint(self, target_pixel_hint):
        if not target_pixel_hint or len(target_pixel_hint) != 2:
            return ""
        hx, hy = target_pixel_hint
        return (
            f"  NOTE: There may be several similar-looking objects (e.g. multiple "
            f"apples) in view. Target specifically the one nearest pixel "
            f"({hx:.0f}, {hy:.0f}) from the previous check -- not just any object "
            f"matching the description."
        )

    def _publish_alignment_result(
        self, object_name, gripper_visible, target_visible,
        offset_x_px, offset_y_px, aligned, confidence, notes,
        gripper_px=None, target_px=None,
    ):
        payload = {
            "object_name": object_name,
            "aligned": aligned,
            "gripper_visible": gripper_visible,
            "target_visible": target_visible,
            "offset_x_px": offset_x_px,
            "offset_y_px": offset_y_px,
            "confidence": confidence,
            "notes": notes,
        }
        if gripper_px is not None:
            payload["gripper_px"] = list(gripper_px)
        if target_px is not None:
            payload["target_px"] = list(target_px)

        msg = String()
        msg.data = json.dumps(payload)
        self.alignment_pub.publish(msg)
        self.get_logger().info(f"Alignment result: {payload}")

    # ------------------------------------------------------------------
    # Shared VLM call
    # ------------------------------------------------------------------
    def _encode_image(self, cv_image):
        success, buffer = cv2.imencode(".jpg", cv_image)
        if not success:
            self.get_logger().error("Failed to encode image as JPEG")
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def query_vlm(self, prompt: str, image_b64: str):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Ollama request failed: {e}")
            return None

        raw_text = response.json().get("response", "").strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            self.get_logger().warning(f"Could not parse VLM response as JSON: {raw_text}")
            return None

    def _backfill_grasp_defaults(self, parsed):
        defaults = {
            "object_name": "unknown object",
            "description": "",
            "estimated_material": "unknown",
            "fragility_score": 5,
            "surface_texture": "unknown",
            "recommended_grip_force": "medium",
            "grasp_method": "power grasp",
            "approach_vector": "top-down",
            "finger_joint_targets": {
                "R_Thumb_Pitch": 0.3, "R_Thumb_Roll": 0.5, "R_Thumb_Yaw": 0.5,
                "R_Thumb_Flexor": 0.3, "R_Thumb_DIP": 0.3,
                "R_Index_Pitch": 0.3, "R_Index_Yaw": 0.5,
                "R_Index_Flexor": 0.3, "R_Index_DIP": 0.3,
                "R_Middle_Pitch": 0.3, "R_Middle_Yaw": 0.5,
                "R_Middle_Flexor": 0.3, "R_Middle_DIP": 0.3,
                "R_Ring_Pitch": 0.3, "R_Ring_Yaw": 0.5,
                "R_Ring_Flexor": 0.3, "R_Ring_DIP": 0.3,
                "R_Pinky_Pitch": 0.3, "R_Pinky_Yaw": 0.5,
                "R_Pinky_Flexor": 0.3, "R_Pinky_DIP": 0.3,
            },
            "confidence": 0.0,
            "notes": "",
            "bbox_center_x": IMAGE_WIDTH_PX // 2,
            "bbox_center_y": IMAGE_HEIGHT_PX // 2,
            "bbox_width_px": 0,
        }
        for key, default_val in defaults.items():
            parsed.setdefault(key, default_val)
        return parsed


def main(args=None):
    rclpy.init(args=args)
    node = VLMOverheadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
