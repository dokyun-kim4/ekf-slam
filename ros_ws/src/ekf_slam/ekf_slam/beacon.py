"""
This file will simulate a beacon that provides range and bearing measurements to the robot.
The beacon will be placed at a fixed location in the environment, and the robot will receive noisy measurements of the range and bearing to the beacon.

We assume the correspondence problem is solved, so each beacon will provide its unique ID with the measurements
Data published here has no noise, as that will be handled by the NoiseInjector node.
"""

import rclpy
from rclpy.node import Node
from ekf_interfaces.msg import BeaconData # type: ignore
from math import atan2

class Beacon():
    def __init__(self, beacon_id, x, y):
        self.id = beacon_id
        self.x = x
        self.y = y
    
    def compute_measurements(self, robot_xy):
        dx = self.x - robot_xy[0]
        dy = self.y - robot_xy[1]
        range_measurement = (dx**2 + dy**2)**0.5
        bearing_measurement = atan2(dy, dx)
        return range_measurement, bearing_measurement

class BeaconNode(Node):
    def __init__(self):
        super().__init__('beacon_node')

        self.beacons = [
            Beacon(1, 5.0, 5.0),  # Example beacon at (5, 5)
            Beacon(2, -5.0, 5.0), # Example beacon at (-5, 5)
            Beacon(3, -5.0, -5.0),# Example beacon at (-5, -5)
        ]
        self.publisher_ = self.create_publisher(BeaconData, 'beacon_measurements', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        pass
    
def main(args=None):
    rclpy.init(args=args)
    beacon_node = BeaconNode()
    rclpy.spin(beacon_node)
    beacon_node.destroy_node()
    rclpy.shutdown()
