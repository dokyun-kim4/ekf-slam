import os

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='ekf_slam',
            executable='beacon'),

        Node(
            package='ekf_slam',
            executable='noise_injector'),
    ])
