#!/bin/bash
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "parameter_bridge" 2>/dev/null
pkill -9 -f "robot_state_publisher" 2>/dev/null
pkill -9 -f "controller_manager" 2>/dev/null
pkill -9 -f "rviz2" 2>/dev/null
sleep 2

sudo rm -f /dev/shm/fastrtps_*
sudo rm -f /dev/shm/sem.fastrtps_*

source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share:~/ur_gz_ws/src/apple_gripper_sim/models

ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=$HOME/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=$HOME/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=$HOME/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world
