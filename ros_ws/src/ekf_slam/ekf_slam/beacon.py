"""
This file will simulate a beacon that provides range and bearing measurements to the robot.
The beacon will be placed at a fixed location in the environment, and the robot will receive noisy measurements of the range and bearing to the beacon.

We assume the correspondence problem is solved, so each beacon will provide its unique ID with the measurements
Data published here has no noise, as that will be handled by the NoiseInjector node.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ekf_interfaces.msg import BeaconData # type: ignore
import tf_transformations

from math import atan2

class Beacon():
    def __init__(self, beacon_id, x, y):
        self.id = beacon_id
        self.x = x
        self.y = y

        self.MAX_RANGE = 1.0
    
    def get_br(self, robot_pose):
        dx = self.x - robot_pose[0]
        dy = self.y - robot_pose[1]
        range_measurement = (dx**2 + dy**2)**0.5
        bearing_measurement = atan2(dy, dx) - robot_pose[2]
        return range_measurement, bearing_measurement, self.id

class BeaconNode(Node):
    def __init__(self):
        super().__init__('beacon_node')

        self.beacon_interval = 3.0  # seconds
        self.beacons = [
            Beacon(1, 5.0, 5.0),  # Example beacon at (5, 5)
            Beacon(2, -5.0, 5.0), # Example beacon at (-5, 5)
            Beacon(3, -5.0, -5.0),# Example beacon at (-5, -5)
        ]
        self.robot_pose = None # x,y,theta

        self.gt_sub = self.create_subscription(Odometry, '/ground_truth' , self.pose_callback, 10)
        self.beacon_pub = self.create_publisher(BeaconData, '/beacon_measurements', 10)
        self.timer = self.create_timer(self.beacon_interval, self.beacon_callback)

    def pose_callback(self,msg: Odometry):
        quat = [msg.pose.pose.orientation.x, 
                msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, 
                msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        # print(self.robot_pose)

    def beacon_callback(self):
        msg = BeaconData()

        if self.robot_pose is None:
            return

        for beacon in self.beacons:
            range, bearing, id = beacon.get_br(self.robot_pose)
            msg.ids.append(id)
            msg.ranges.append(range)
            msg.bearings.append(bearing)
            self.beacon_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    beacon_node = BeaconNode()
    rclpy.spin(beacon_node)
    beacon_node.destroy_node()
    rclpy.shutdown()
