FROM osrf/ros:humble-desktop-full-jammy

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
COPY .devcontainer/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

RUN sudo apt-get update && sudo apt-get install -y \
ros-humble-rqt \
ros-humble-rqt-common-plugins \
ros-humble-rviz2 \
ros-humble-gazebo-ros-pkgs \
ros-humble-gazebo-plugins \
ros-humble-gazebo-msgs \
gazebo \
mesa-utils

# Update apt sources
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg