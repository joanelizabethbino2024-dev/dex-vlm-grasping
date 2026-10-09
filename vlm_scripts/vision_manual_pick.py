#!/usr/bin/env python3
"""
Vision-Triggered Manual Pick
------------------------------
Waits for one real detection from yolo_depth_detector.py (real depth-camera
3D position, not a hardcoded/typed-in one), then hands that position
straight to manual_target_node.py's ManualTargetNode -- the proven,
live-tested pick sequence for this exact robot: approach with the measured
wrist-to-fingertip offset, pre-curl, descend with REAL contact detection
(commanded-vs-actual joint position error, not a blind timer), thumb nudge,
full close, then lift to test the grip. All of that logic is reused
as-is; this script only supplies the target position.

Usage: python3 vision_manual_pick.py [--close-hand 0.6]
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from manual_target_node import ManualTargetNode

FRAGILITY_TOPIC = "/overhead_camera/fragility_analysis"


class DetectionWaiter(Node):
    def __init__(self):
        super().__init__("vision_manual_pick_waiter")
        self.target = None
        self.create_subscription(String, FRAGILITY_TOPIC, self.on_detection, 10)

    def on_detection(self, msg: String):
        if self.target is not None:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if data.get("object_name") == "none visible":
            return
        if "base_link_x" not in data or "base_link_y" not in data or "base_link_z" not in data:
            return  # detector without real depth (e.g. VLM-only) -- wait for a real one
        self.target = (data["base_link_x"], data["base_link_y"], data["base_link_z"])
        self.get_logger().info(f"Got real depth-based target: {self.target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-hand", type=float, default=0.6)
    parser.add_argument("--hand-delay", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()

    waiter = DetectionWaiter()
    print(f"Waiting for a real detection on {FRAGILITY_TOPIC} "
          "(make sure yolo_depth_detector.py is running)...", flush=True)
    while waiter.target is None:
        rclpy.spin_once(waiter, timeout_sec=0.5)
    target = waiter.target
    waiter.destroy_node()

    print(f"Triggering the proven pick sequence at base_link {target}, "
          f"close_hand={args.close_hand}", flush=True)
    node = ManualTargetNode(target, "base_link", args.close_hand, args.hand_delay)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
