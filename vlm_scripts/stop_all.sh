#!/bin/bash
# Kills the sim and every pipeline node (kept in a file so the patterns aren't
# in the calling shell's own command line).
for pat in "ign gazebo" yolo_depth_detector.py vision_ik_overhead_closed_loop.py finger_alignment_node.py vlm_overhead_node.py "ros2 launch" launch_all.sh parameter_bridge robot_state_publisher; do
  ps aux | grep -F -- "$pat" | grep -v grep | grep -v stop_all.sh | awk '{print $2}' | xargs -r kill -9
done
