

```bash
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "parameter_bridge" 2>/dev/null
sleep 2
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share:~/ur_gz_ws/src/apple_gripper_sim/models

ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=/home/tt501/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world
```
```bash

source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
ros2 run controller_manager spawner dexhand_controller --controller-manager /controller_manager
```
```bash
to execute the project with vlm

 ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e description_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro controllers_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml world_file:=/home/tt501/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world

```
```bash
ros2 run ros_gz_bridge parameter_bridge /gripper_camera@sensor_msgs/msg/Image[ignition.msgs.Image

```

```bash
cd ~/vlm_scripts && source /opt/ros/humble/setup.bash && python3 vlm_fragility_node.py

```
```bash

ros2 topic echo /gripper_camera/fragility_analysis

```
