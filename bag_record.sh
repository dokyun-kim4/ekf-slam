#!/bin/bash

cd /ekf-slam/ros_ws
. install/setup.bash
ros2 bag record -o test2 /ground_truth /beacon_ground_truth /beacon_noisy /encoder_noisy /gps_noisy /robot_description /tf /tf_static
