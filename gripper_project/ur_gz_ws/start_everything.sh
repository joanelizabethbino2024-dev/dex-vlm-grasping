#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share

echo "=== Killing any leftover processes ==="
pkill -9 -f "ign gazebo server" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "ros2 launch" 2>/dev/null || true
sleep 2

echo "=== Launching simulation (background) ==="
setsid ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    > /tmp/sim_launch.log 2>&1 < /dev/null &
disown

echo "Waiting for simulation to be ready..."
sleep 12

echo "=== Spawning red block ==="
ros2 run ros_gz_sim create -world empty -file ~/ur_gz_ws/src/my_pick_and_place/urdf/red_block.sdf -name red_block -x 0.5 -y 0.0 -z 0.1

echo "=== Activating hand controller ==="
ros2 run controller_manager spawner dexhand_controller --controller-manager /controller_manager

echo "=== Starting pose bridge (background) ==="
ros2 run ros_gz_bridge parameter_bridge /model/red_block/pose@geometry_msgs/msg/Pose[ignition.msgs.Pose > /tmp/pose_bridge.log 2>&1 &

echo "=== Starting attach/detach bridge (background) ==="
ros2 run ros_gz_bridge parameter_bridge /attach@std_msgs/msg/Empty]ignition.msgs.Empty /detach@std_msgs/msg/Empty]ignition.msgs.Empty > /tmp/attach_bridge.log 2>&1 &

sleep 3
echo ""
echo "=== SETUP COMPLETE. Verifying... ==="
ros2 control list_controllers
echo ""
ign topic -l | grep -i attach
echo ""
echo "If you see dexhand_controller ACTIVE and /attach listed above, everything is ready."
echo "Now run: python3 ~/ur_gz_ws/src/my_pick_and_place/scripts/pick_and_place.py"
