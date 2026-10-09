#!/usr/bin/env python3
import base64
import json
import math
import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"
IMAGE_TOPIC = "/overhead_camera"
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
HORIZONTAL_FOV_RAD = 1.396
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(HORIZONTAL_FOV_RAD / 2.0)
CAMERA_WORLD_X = 1.125
CAMERA_WORLD_Y = 0.0
CAMERA_WORLD_Z = 2.0
APPLE_WORLD_Z = 0.05
U_SIGN = 1.0
V_SIGN = 1.0
WORLD_FRAME = "base_footprint"
BASE_FRAME = "base_link"

PROMPT = """Look at this overhead image of a robot workspace. There is a single
apple sitting on a surface below a robot arm. Ignore the robot arm, gripper,
and any mounting hardware -- only report on the apple itself.

Respond with ONLY a valid JSON object (no markdown, no extra text) with these
exact keys:

{
  "object_name": "apple",
  "description": "one short sentence describing the apple's color, size, condition",
  "bbox_center_x": 0-640 integer, horizontal pixel coordinate of the apple's center,
  "bbox_center_y": 0-480 integer, vertical pixel coordinate of the apple's center,
  "bbox_width_px": integer, approximate width in pixels of the apple,
  "confidence": 0.0-1.0 float
}

If no apple is visible, set object_name to "none visible" and the pixel
fields to 0.

Respond with ONLY the JSON object, nothing else."""

class QueryApplePixelNode(Node):
    def __init__(self):
        super().__init__("query_apple_vlm_node")
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.got_frame = False
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.get_logger().info(f"Waiting for one frame on {IMAGE_TOPIC}...")

    def image_callback(self, msg):
        if self.got_frame:
            return
        self.got_frame = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            rclpy.shutdown()
            return
        print(f"\nFrame received. Querying {MODEL_NAME}... (this can take up to ~90s on CPU)\n")
        success, buffer = cv2.imencode(".jpg", cv_image)
        if not success:
            self.get_logger().error("Failed to encode image.")
            rclpy.shutdown()
            return
        image_b64 = base64.b64encode(buffer).decode("utf-8")
        analysis = self.query_vlm(image_b64)
        if analysis is None:
            rclpy.shutdown()
            return
        self.print_results(analysis)
        rclpy.shutdown()

    def query_vlm(self, image_b64):
        payload = {"model": MODEL_NAME, "prompt": PROMPT, "images": [image_b64], "stream": False, "format": "json"}
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}")
            return None
        raw_text = response.json().get("response", "").strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"Could not parse VLM response as JSON:\n{raw_text}")
            return None

    def pixel_to_world(self, px, py):
        camera_height_above_apple = CAMERA_WORLD_Z - APPLE_WORLD_Z
        m_per_px = camera_height_above_apple / FOCAL_LENGTH_PX
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
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = world_point
            transform = self.tf_buffer.lookup_transform(BASE_FRAME, WORLD_FRAME, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            transformed = do_transform_point(point_stamped, transform)
            return (transformed.point.x, transformed.point.y, transformed.point.z)
        except Exception as e:
            print(f"TF transform failed: {e}")
            return None

    def print_results(self, analysis):
        print("=" * 70)
        print("VLM RAW ANALYSIS")
        print("=" * 70)
        for key, val in analysis.items():
            print(f"  {key}: {val}")
        obj = analysis.get("object_name", "unknown")
        if obj == "none visible":
            print("\nNo apple detected in this frame.")
            return
        px = analysis.get("bbox_center_x", IMAGE_WIDTH_PX / 2)
        py = analysis.get("bbox_center_y", IMAGE_HEIGHT_PX / 2)
        world_point = self.pixel_to_world(px, py)
        print("\n" + "=" * 70)
        print("CONVERTED COORDINATES")
        print("=" * 70)
        print(f"  Pixel (from VLM):        ({px}, {py})")
        print(f"  World frame ({WORLD_FRAME}):  ({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f})")
        base_point = self.transform_world_to_base(world_point)
        if base_point:
            print(f"  {BASE_FRAME} frame:            ({base_point[0]:.3f}, {base_point[1]:.3f}, {base_point[2]:.3f})")
        print("=" * 70)
        print("\nCompare the World frame line above against the apple's real pose from Gazebo.")

def main():
    rclpy.init()
    node = QueryApplePixelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
