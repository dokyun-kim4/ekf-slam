"""
This file will simulate a beacon that provides range and bearing measurements to the robot.
The beacon will be placed at a fixed location in the environment, and the robot will receive noisy measurements of the range and bearing to the beacon.

We assume the correspondence problem is solved, so each beacon will provide its unique ID with the measurements
Data published here has no noise, as that will be handled by the NoiseInjector node.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from ekf_interfaces.msg import BeaconData
import tf_transformations
from gazebo_msgs.srv import SpawnEntity

from math import atan2
import random
from typing import List, cast

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

class Beacon():
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
        range_measurement = (dx**2 + dy**2)**0.5
        bearing_measurement = atan2(dy, dx) - robot_pose[2]
        return range_measurement, bearing_measurement, self.id

class BeaconNode(Node):
    """
    ROS 2 node that manages a collection of simulated beacons.
    Handles visually spawning them in Gazebo and publishing the ground truth 
    range and bearing measurements relative to the robot's pose.
    """
    def __init__(self):
        """Initializes the node, defines the beacons, and sets up publishers/subscribers."""
        # Allow reading arbitrarily structured YAML files dynamically 
        super().__init__('beacon_node', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.beacon_ids = cast(List[int], self.get_parameter_or("beacon_ids", Parameter("beacon_ids", Parameter.Type.INTEGER_ARRAY, [1, 2, 3])).value)
        self.beacon_x_pos = cast(List[float], self.get_parameter_or("beacon_x_pos", Parameter("beacon_x_pos", Parameter.Type.DOUBLE_ARRAY, [1.0, -1.0, -2.0])).value)
        self.beacon_y_pos = cast(List[float], self.get_parameter_or("beacon_y_pos", Parameter("beacon_y_pos", Parameter.Type.DOUBLE_ARRAY, [1.0, 1.5, 2.0])).value)
        self.RANGE_NOISE = cast(float, self.get_parameter_or("range_noise", Parameter("range_noise", Parameter.Type.DOUBLE, 0.1)).value)
        self.RANGE_PROP_NOISE = cast(float, self.get_parameter_or("range_prop_noise", Parameter("range_prop_noise", Parameter.Type.DOUBLE, 0.05)).value)
        self.BEARING_NOISE = cast(float, self.get_parameter_or("bearing_noise", Parameter("bearing_noise", Parameter.Type.DOUBLE, 0.08)).value)
        self.MAX_RANGE = cast(float, self.get_parameter_or("max_range", Parameter("max_range", Parameter.Type.DOUBLE, 2.5)).value)
        self.INTERVAL = cast(float, self.get_parameter_or("interval", Parameter("interval", Parameter.Type.DOUBLE, 2.0)).value)

        beacon_info = zip(self.beacon_ids, self.beacon_x_pos, self.beacon_y_pos)
        self.beacons = [Beacon(beacon_id, x, y) for beacon_id, x, y in beacon_info]

        self.robot_pose = None # x,y,theta

        # Setup spawn entity client
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.spawn_beacons()

        self.gt_sub = self.create_subscription(Odometry, '/ground_truth' , self.pose_callback, 10)
        self.beacon_gt_pub = self.create_publisher(BeaconData, '/beacon_ground_truth', 10)
        self.beacon_noisy_pub = self.create_publisher(BeaconData, '/beacon_noisy', 10)
        self.timer = self.create_timer(self.INTERVAL, self.beacon_callback)

    def spawn_beacons(self):
        """
        Requests Gazebo to spawn a visual cylinder model for each beacon.
        Cycles through a predefined list of colors.
        """
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Gazebo /spawn_entity service not available. Beacons will not be visually spawned.')
            return

        colors = ["Gazebo/Red", "Gazebo/Blue", "Gazebo/Green", "Gazebo/Yellow", "Gazebo/Purple", "Gazebo/Orange"]
        
        for i, beacon in enumerate(self.beacons):
            req = SpawnEntity.Request()
            name = f'beacon_{beacon.id}'
            req.name = name
            color = colors[i % len(colors)]
            req.xml = BEACON_SDF.format(name=name, color=color)
            req.robot_namespace = ""
            
            req.initial_pose.position.x = beacon.x
            req.initial_pose.position.y = beacon.y
            req.initial_pose.position.z = 0.5 
            
            self.spawn_client.call_async(req)
            self.get_logger().info(f"Requested visual spawn for {name} at ({beacon.x}, {beacon.y})")

    def pose_callback(self,msg: Odometry):
        """
        Updates the internal robot pose based on ground truth odometry.

        Args:
            msg (Odometry): The incoming odometry message containing ground truth pose.
        """
        quat = [msg.pose.pose.orientation.x, 
                msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, 
                msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def beacon_callback(self):
        """
        Periodically computes and publishes the theoretical range and bearing 
        from the robot to all known beacons.
        """
        gt_msg = BeaconData()
        noisy_msg = BeaconData()

        if self.robot_pose is None:
            return

        for beacon in self.beacons:
            range, bearing, id = beacon.get_br(self.robot_pose)

            # Publish ground truth for logging
            gt_msg.ids.append(id)
            gt_msg.ranges.append(range)
            gt_msg.bearings.append(bearing)
            gt_msg.x_poses.append(beacon.x)
            gt_msg.y_poses.append(beacon.y)
            self.beacon_gt_pub.publish(gt_msg)

            # Publish noisy measurements, also skip beacons that are out of range
            if range > self.MAX_RANGE:
                continue
            noisy_msg.ids.append(id)
            noisy_msg.ranges.append(random.gauss(range, self.RANGE_NOISE + self.RANGE_PROP_NOISE * range))
            noisy_msg.bearings.append(random.gauss(bearing, self.BEARING_NOISE))
            noisy_msg.x_poses.append(beacon.x)
            noisy_msg.y_poses.append(beacon.y)
            self.beacon_noisy_pub.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    beacon_node = BeaconNode()
    rclpy.spin(beacon_node)
    beacon_node.destroy_node()
    rclpy.shutdown()
