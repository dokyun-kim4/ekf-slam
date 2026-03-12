import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    pkg_dir = get_package_share_directory("ekf_slam")
    rviz_config_file = os.path.join(pkg_dir, "config.rviz")
    ekf_config_file = os.path.join(pkg_dir, "config.yaml")
    return LaunchDescription(
        [
            Node(
                package="ekf_slam",
                executable="path_viz",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="ekf_slam",
                executable="ekf_slam",
                arguments=["--ros-args", "--params-file", ekf_config_file],
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config_file],
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
