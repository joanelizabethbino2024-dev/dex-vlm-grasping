import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('apple_gripper_sim')
    models_path = os.path.join(pkg_share, 'models')
    world_path = os.path.join(pkg_share, 'worlds', 'apple_world.world')

    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    gazebo_launch = os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')

    # Make sure Gazebo can resolve model://apple_XX URIs used in the world file
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    new_model_path = models_path + (os.pathsep + existing_model_path if existing_model_path else '')

    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', new_model_path)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={'world': world_path, 'verbose': 'true'}.items(),
    )

    return LaunchDescription([
        set_model_path,
        gazebo,
    ])
