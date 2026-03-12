"""
Utility functions and classes for the EKF SLAM implementation.
"""

import numpy as np
from math import atan2

BEACON_SDF = """
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>{color}</name>
          </script>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

class Beacon:
    """
    Represents a single static beacon in the environment.
    """

    def __init__(self, beacon_id, x, y):
        """
        Initialize the beacon.

        Args:
            beacon_id (int): Unique identifier for the beacon.
            x (float): The x-coordinate of the beacon's location.
            y (float): The y-coordinate of the beacon's location.
        """
        self.id = beacon_id
        self.x = x
        self.y = y

    def get_br(self, robot_pose):
        """
        Calculate the exact range and bearing to the beacon from the given robot pose.

        Args:
            robot_pose (list): The full pose of the robot as [x, y, theta].

        Returns:
            tuple: A tuple containing (range, bearing, beacon_id).
        """
        dx = self.x - robot_pose[0]
        dy = self.y - robot_pose[1]
        range_measurement = (dx**2 + dy**2) ** 0.5
        bearing_measurement = atan2(dy, dx) - robot_pose[2]
        return range_measurement, bearing_measurement, self.id



def wrap_angle(angle: float) -> float:
        """
        Wraps the given angle to the range [-pi, pi].

        Args:
            angle (float): The angle in radians

        Returns:
            float: The wrapped angle in radians
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
