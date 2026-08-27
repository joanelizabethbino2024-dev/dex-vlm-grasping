#!/usr/bin/env bash
# Alternative to the world-file method: spawns all 10 apples one-by-one via the
# gazebo_ros spawn_entity.py node into an ALREADY-RUNNING empty Gazebo world.
#
# Usage (in a sourced ROS 2 + workspace terminal, with Gazebo already running,
# e.g. from: ros2 launch gazebo_ros gazebo.launch.py):
#
#   chmod +x spawn_all_apples.sh
#   ./spawn_all_apples.sh
#
# Requires: GAZEBO_MODEL_PATH already contains the apple_gripper_sim/models dir
# (the main launch file sets this automatically; if spawning manually, export
# it yourself first, see README).

set -e

APPLES=(apple_01 apple_02 apple_03 apple_04 apple_05 apple_06 apple_07 apple_08 apple_09 apple_10)
XS=(0.00 0.15 0.30 0.45 0.60 0.00 0.15 0.30 0.45 0.60)
YS=(0.00 0.00 0.00 0.00 0.00 0.15 0.15 0.15 0.15 0.15)
Z=0.05

PKG_SHARE=$(ros2 pkg prefix apple_gripper_sim)/share/apple_gripper_sim
MODELS_DIR="${PKG_SHARE}/models"

for i in "${!APPLES[@]}"; do
  name="${APPLES[$i]}"
  x="${XS[$i]}"
  y="${YS[$i]}"
  sdf_file="${MODELS_DIR}/${name}/model.sdf"

  echo ">>> Spawning ${name} at (${x}, ${y}, ${Z})"
  ros2 run gazebo_ros spawn_entity.py \
    -entity "${name}" \
    -file "${sdf_file}" \
    -x "${x}" -y "${y}" -z "${Z}"
done

echo "All 10 apples spawned."
