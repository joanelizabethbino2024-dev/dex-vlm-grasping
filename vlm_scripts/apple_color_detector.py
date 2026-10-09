#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import json
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

IMAGE_TOPIC = "/overhead_camera"
RESULT_TOPIC = "/overhead_camera/fragility_analysis"

IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
DETECT_INTERVAL_SEC = 0.5

SATURATION_MIN = 80
VALUE_MIN = 40
VALUE_MAX = 250

MIN_BLOB_AREA_PX = 30
MAX_BLOB_AREA_PX = 5000
MIN_CIRCULARITY = 0.5

DEFAULT_FINGER_TARGETS = {
    "R_Thumb_Pitch": 0.4, "R_Thumb_Roll": 0.5, "R_Thumb_Yaw": 0.5,
    "R_Thumb_Flexor": 0.4, "R_Thumb_DIP": 0.4,
    "R_Index_Pitch": 0.4, "R_Index_Yaw": 0.5, "R_Index_Flexor": 0.4, "R_Index_DIP": 0.4,
    "R_Middle_Pitch": 0.4, "R_Middle_Yaw": 0.5, "R_Middle_Flexor": 0.4, "R_Middle_DIP": 0.4,
    "R_Ring_Pitch": 0.2, "R_Ring_Yaw": 0.5, "R_Ring_Flexor": 0.2, "R_Ring_DIP": 0.2,
    "R_Pinky_Pitch": 0.2, "R_Pinky_Yaw": 0.5, "R_Pinky_Flexor": 0.2, "R_Pinky_DIP": 0.2,
}

class AppleColorDetectorNode(Node):
    def __init__(self):
        super().__init__("apple_color_detector_node")
        self.bridge = CvBridge()
        self.last_detect_time = 0.0
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 10)
        self.get_logger().info(f"Subscribed to {IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing detections to {RESULT_TOPIC}")
        self.get_logger().info("Using OpenCV color/contour detection (no VLM).")

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_detect_time < DETECT_INTERVAL_SEC:
            return
        self.last_detect_time = now
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return
        blobs = self.detect_blobs(cv_image)
        if not blobs:
            result = {
                "object_name": "none visible", "description": "", "fragility_score": 0,
                "finger_joint_targets": {k: 0.0 for k in DEFAULT_FINGER_TARGETS},
                "confidence": 0.0, "notes": "no colorful round blobs detected",
                "bbox_center_x": IMAGE_WIDTH_PX // 2, "bbox_center_y": IMAGE_HEIGHT_PX // 2,
                "bbox_width_px": 0,
            }
        else:
            cx, cy = IMAGE_WIDTH_PX / 2, IMAGE_HEIGHT_PX / 2
            best = min(blobs, key=lambda b: math.hypot(b["cx"] - cx, b["cy"] - cy))
            result = {
                "object_name": "apple",
                "description": f"detected apple, ~{best['width_px']}px wide",
                "fragility_score": 4, "finger_joint_targets": DEFAULT_FINGER_TARGETS,
                "confidence": 1.0, "notes": f"{len(blobs)} candidate blob(s) found this frame",
                "bbox_center_x": int(best["cx"]), "bbox_center_y": int(best["cy"]),
                "bbox_width_px": int(best["width_px"]),
            }
        msg_out = String()
        msg_out.data = json.dumps(result)
        self.result_pub.publish(msg_out)
        self.get_logger().info(f"Detection: {result}")

    def detect_blobs(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, SATURATION_MIN, VALUE_MIN])
        upper = np.array([179, 255, VALUE_MAX])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA_PX or area > MAX_BLOB_AREA_PX:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(c)
            blobs.append({"cx": cx, "cy": cy, "width_px": max(w, h), "area": area})
        return blobs

def main(args=None):
    rclpy.init(args=args)
    node = AppleColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
