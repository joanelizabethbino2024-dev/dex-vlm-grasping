#!/usr/bin/env python3
import base64
import json
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
IMAGE_TOPIC = "/gripper_camera"
RESULT_TOPIC = "/gripper_camera/fragility_analysis"
QUERY_INTERVAL_SEC = 2.0

# Joint angles are normalized 0.0 (fully open/straight) to 1.0 (fully curled/closed).
# This maps cleanly onto a joint_trajectory_controller by scaling to each joint's
# real min/max radians in your controller node.
PROMPT = """You are a robotic grasping assistant for a 5-finger dexterous robot hand
(thumb, index, middle, ring, pinky), each finger with a knuckle, middle, and tip joint.
A gripper-mounted camera sees an object. Analyze the image and decide how to pick it up safely.

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
    "R_Thumb_Pitch": 0.0-1.0,
    "R_Thumb_Roll": 0.0-1.0,
    "R_Thumb_Yaw": 0.0-1.0,
    "R_Thumb_Flexor": 0.0-1.0,
    "R_Thumb_DIP": 0.0-1.0,
    "R_Index_Pitch": 0.0-1.0,
    "R_Index_Yaw": 0.0-1.0,
    "R_Index_Flexor": 0.0-1.0,
    "R_Index_DIP": 0.0-1.0,
    "R_Middle_Pitch": 0.0-1.0,
    "R_Middle_Yaw": 0.0-1.0,
    "R_Middle_Flexor": 0.0-1.0,
    "R_Middle_DIP": 0.0-1.0,
    "R_Ring_Pitch": 0.0-1.0,
    "R_Ring_Yaw": 0.0-1.0,
    "R_Ring_Flexor": 0.0-1.0,
    "R_Ring_DIP": 0.0-1.0,
    "R_Pinky_Pitch": 0.0-1.0,
    "R_Pinky_Yaw": 0.0-1.0,
    "R_Pinky_Flexor": 0.0-1.0,
    "R_Pinky_DIP": 0.0-1.0
  },
  "confidence": 0.0-1.0 float representing how confident you are in this analysis,
  "notes": "one short sentence with any additional relevant observation",
  "bbox_center_x": 0-640 integer, horizontal pixel coordinate of the object's center (0=left edge, 640=right edge),
  "bbox_center_y": 0-480 integer, vertical pixel coordinate of the object's center (0=top edge, 480=bottom edge),
  "bbox_width_px": integer, approximate width in pixels of the object as it appears in the image
}

Guidance for finger_joint_targets (this hand's real joint names, one flat dict, all values normalized):
- Each non-thumb finger (Index, Middle, Ring, Pinky) has 4 joints: Pitch (base curl),
  Yaw (side-to-side spread), Flexor (main curl actuator), and DIP (fingertip curl).
- Thumb has 5 joints: Pitch, Roll, Yaw, Flexor, DIP (extra Roll joint for opposability).
- 0.0 means that joint is fully open/straight/neutral, 1.0 means fully curled/closed
  (for Yaw and Roll joints, 0.5 is neutral/centered, 0.0/1.0 are the two extremes).
- For fragile or soft objects: use lower Flexor/DIP values (light closure) and prefer a
  tripod grasp (Thumb + Index + Middle actively closing; Ring/Pinky Flexor/DIP near 0).
- For rigid or heavy objects: use a fuller power grasp with higher Flexor/DIP closure
  across all five fingers.
- Set Thumb_Roll and Thumb_Yaw so the thumb opposes the other fingers for a stable grasp.
- If no object is clearly visible, set object_name to "none visible", fragility_score to 0,
  and all finger_joint_targets to 0.0 (0.5 for any Yaw/Roll joints).

The camera image is 640x480 pixels. Estimate bbox_center_x, bbox_center_y, and
bbox_width_px as accurately as you can from what you see -- these are used to
physically position the robot arm, so a rough estimate is far better than
omitting them.

Respond with ONLY the JSON object, nothing else."""


class VLMFragilityNode(Node):
    def __init__(self):
        super().__init__("vlm_fragility_node")
        self.bridge = CvBridge()
        self.last_query_time = 0.0
        self.subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, 10
        )
        self.publisher = self.create_publisher(String, RESULT_TOPIC, 10)
        self.get_logger().info(f"Subscribed to {IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing analysis to {RESULT_TOPIC}")
        self.get_logger().info(f"Using Ollama model: {MODEL_NAME}")

    def image_callback(self, msg: Image):
        now = time.time()
        if now - self.last_query_time < QUERY_INTERVAL_SEC:
            return
        self.last_query_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Sanity check: warn if the frame looks basically empty/black, since a blank
        # image is the most common reason the VLM keeps returning "unknown object".
        mean_val = cv_image.mean()
        if mean_val < 5.0:
            self.get_logger().warning(
                f"Incoming frame looks almost black (mean pixel={mean_val:.2f}). "
                "Check camera pose/lighting in the world file."
            )

        success, buffer = cv2.imencode(".jpg", cv_image)
        if not success:
            self.get_logger().error("Failed to encode image as JPEG")
            return

        image_b64 = base64.b64encode(buffer).decode("utf-8")
        analysis = self.query_vlm(image_b64)

        if analysis is not None:
            result_msg = String()
            result_msg.data = json.dumps(analysis)
            self.publisher.publish(result_msg)
            self.get_logger().info(f"Analysis: {analysis}")

    def query_vlm(self, image_b64: str):
        payload = {
            "model": MODEL_NAME,
            "prompt": PROMPT,
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
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            self.get_logger().warning(f"Could not parse VLM response as JSON: {raw_text}")
            return {"raw_response": raw_text}

        # Backfill missing keys so downstream consumers can rely on a stable schema
        # even if the model omits a field on a given call.
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
            "bbox_center_x": 320,
            "bbox_center_y": 240,
            "bbox_width_px": 0,
        }
        for key, default_val in defaults.items():
            parsed.setdefault(key, default_val)

        return parsed


def main(args=None):
    rclpy.init(args=args)
    node = VLMFragilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
