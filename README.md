# EKF-SLAM Implementation with ROS2 and Gazebo
**By Dokyun Kim**

This repository contains an implementation of the EKF SLAM algorithm using ROS2 and Gazebo. The learning objectives of this project was to improve understanding of the mathematical foundations of the EKF SLAM algorithm, as well as to gain practical experience in implementing it using popular tools like ROS2 and Gazebo. The simulation environment is built on top of the [Neato simulation](https://github.com/comprobo23/neato_packages) from Olin's CompRobo course.


## Setup
To set up the project, follow these steps:  

#1. Clone the repository and navigate to the project directory:
```bash
git clone <repo url>
cd ekf-slam
```

#2. The project uses a ROS2 devcontainer for development. We recommend using VS Code's Remote - Containers extension to work with the devcontainer. Open the project in VS Code and follow the prompts to reopen the project in the container.

#3. Once you have the devcontainer set up, you can build the ROS2 workspace:
```bash
# Aliases are defined for commonly used ros2 commands
rb # Builds and sources the ROS2 workspace
```

## Running the algorithm
The EKF SLAM node can either be run live in the Gazebo simulation or with pre-recorded ROS2 bag files.


### Running with Gazebo sim
```bash
i # Sources the ROS2 workspace
sim # Runs the Gazebo world
```
```bash
i # Sources the ROS2 workspace
ros2 launch ekf_slam ekf_slam.launch.py # Launches the EKF SLAM node with appropriate sensors and viz nodes
```

```bash
i # Sources the ROS2 workspace
ros2 run teleop_twist_keyboard teleop_twist_keyboard # Runs the teleop node to control the robot with keyboard inputs
```

Now the robot can be controlled with the keyboard and the EKF SLAM algorithm will run in real-time, estimating the robot's pose and mapping the environment.

<!-- [![EKF SLAM demo](http://img.youtube.com/vi/q7uZBwsd4Ug/0.jpg)](http://www.youtube.com/watch?v=q7uZBwsd4Ug "EKF SLAM demo") -->

### Running with a ROS2 bag file

To record a ROS2 bag file, run the following commands in separate terminals:
```bash
i
sim
```
```bash
ros2 launch ekf_slam record_rosbag.launch
```
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
When you are ready to start recording, run the following script and start driving the robot around.
```
cd /ekf-slam
./bag_record.sh <bag_name>
```
Press Ctrl + C to stop recording; This will store the bag file in `/ekf-slam/ros_ws/rosbag`. To run the EKF SLAM algorithm with the recorded bag file, use the following command:
```bash
ros2 launch ekf_slam demo_with_bag.launch.py
```
```
ros2 bag play rosbag/<bag_name>
