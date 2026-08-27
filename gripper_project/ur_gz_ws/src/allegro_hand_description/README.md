The Allegro Hand Robot Description package contains XACRO and URDF models of Allegro V4 hand. You can use this package to visualise and integrate the hand with your robots. We support [ROS1](https://github.com/touchlab-avatarx/allegro_hand_description/tree/ros1) and [ROS2](https://github.com/touchlab-avatarx/allegro_hand_description).

For examples of how to integrate each device look at `urdf/*.urdf.xacro` files. These attach
the device to a fixed world link for visualisation. Each device is defined as a XACRO
in the corresponding `*.xacro` file. When included a macro for attaching the device to
the robot will be defined. Each macro has a prefix (for uniquely naming all added links and joints),
parent (parent link to attach the device to), and origin (relative offset block).

To see the device in RViz, see example launch
files in `launch/`.
