import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from ekf_interfaces.msg import BeaconData
import random
from typing import Optional

class NoiseInjector(Node):
    def __init__(self):
        super().__init__('noise_injector')

        # --- Sensor Parameters --- #
        self.MOTOR_LINEAR_NOISE = 0.1 # m/s
        self.MOTOR_ANGULAR_NOISE = 0.05 # rad/s

        self.GPS_X_NOISE = 0.1 # m
        self.GPS_Y_NOISE = 0.1 # m
        self.GPS_INTERVAl = 1.0 # seconds

        # TODO: Just inject noise in beacon node and publish both gt and noisy there. 
        self.BEACON_RANGE_NOISE = 0.1 # m
        self.BEACON_RANGE_PROP_NOISE = 0.05 # m
        self.BEACON_BEARING_NOISE = 0.08 # rad
        self.BEACON_INTERVAL = 2.0 # seconds
        self.BEACON_MAX_RANGE = 2.5 # m

        self.ENCODER_LINEAR_NOISE = 0.05 # m/s
        self.ENCODER_LINEAR_NOISE_RATIO = 0.01 # m/s
        self.ENCODER_ANGULAR_NOISE = 0.1 # rad/s
        self.ENCODER_ANGULAR_NOISE_RATIO = 0.1 # rad/s
        self.ENCODER_INTERVAL = 0.2 # seconds

        # --- Stored Ground Truth State --- #
        self.latest_cmd_vel_gt: Optional[Twist] = None
        self.latest_cmd_vel_actual: Optional[Twist] = None # need this for encoder
        self.latest_gps_gt: Optional[Odometry] = None
        self.latest_beacon_gt: Optional[BeaconData] = None

        # --- Subscribers, Publishers, and Timers --- #

        self.gps_timer = self.create_timer(self.GPS_INTERVAl, self.make_noisy_gps)
        self.beacon_timer = self.create_timer(self.BEACON_INTERVAL, self.make_noisy_beacon)
        self.encoder_timer = self.create_timer(self.ENCODER_INTERVAL, self.make_noisy_encoder)

        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.gps_sub = self.create_subscription(Odometry, '/ground_truth', self.gps_callback, 10)
        self.beacon_sub = self.create_subscription(BeaconData, '/beacon_measurements', self.beacon_callback, 10)

        self.cmd_vel_noisy_pub = self.create_publisher(Twist, '/cmd_vel_noisy', 10)
        self.gps_noisy_pub = self.create_publisher(Pose2D, '/gps_noisy', 10)
        self.beacon_noisy_pub = self.create_publisher(BeaconData, '/beacon_noisy', 10)
        self.encoder_vel_pub = self.create_publisher(Twist, '/encoder_vel', 10)

    # --- Simple Timer Callbacks to update states --- #
    def gps_callback(self, msg: Odometry):
        self.latest_gps_gt = msg
    
    def beacon_callback(self, msg: BeaconData):
        self.latest_beacon_gt = msg
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Saves the latest ground truth `/cmd_vel', and immediately publishes a noisy version to `/cmd_vel_noisy`.
        Also saves the latest executed cmd_vel (with noise) for use in the encoder.

        Args:
            msg (Twist): `/cmd_vel` topic
        """
        self.latest_cmd_vel_gt = msg
        noisy_cmd = Twist()
        # Our lin_vel command is in X axis since it is a differential drive robot
        noisy_cmd.linear.x = msg.linear.x + random.gauss(0, self.MOTOR_LINEAR_NOISE)
        # Similarly, our ang_vel command is in Z axis
        noisy_cmd.angular.z = msg.angular.z + random.gauss(0, self.MOTOR_ANGULAR_NOISE)
        self.latest_cmd_vel_actual = noisy_cmd
        self.cmd_vel_noisy_pub.publish(noisy_cmd)


    # --- Noise injection for spoofed sensor data--- #

    def make_noisy_gps(self):
        """
        Simulate a noisy GPS reading based on the latest ground truth GPS position.
        """
        if self.latest_gps_gt is None:
            return
        noisy_msg = Pose2D()
        noisy_msg.x = self.latest_gps_gt.pose.pose.position.x + random.gauss(0, self.GPS_X_NOISE)
        noisy_msg.y = self.latest_gps_gt.pose.pose.position.y + random.gauss(0, self.GPS_Y_NOISE)
        self.gps_noisy_pub.publish(noisy_msg)

    def make_noisy_encoder(self):
        """
        Simulate noisy encoder readings based on the latest actual cmd_vel (which already has motor noise). 
        """
        if self.latest_cmd_vel_actual is None:
            return
        encoder_vel = Twist()
        encoder_vel.linear.x = random.gauss(
                                            self.latest_cmd_vel_actual.linear.x, 
                                            self.ENCODER_LINEAR_NOISE + abs(self.latest_cmd_vel_actual.linear.x)*self.ENCODER_LINEAR_NOISE_RATIO
                                            )
        encoder_vel.angular.z = random.gauss(
                                            self.latest_cmd_vel_actual.angular.z, 
                                            self.ENCODER_ANGULAR_NOISE + abs(self.latest_cmd_vel_actual.angular.z)*self.ENCODER_ANGULAR_NOISE_RATIO
                                            )
        self.encoder_vel_pub.publish(encoder_vel)
    
    def make_noisy_beacon(self):
        """
        Simulate noisy beacon readings based on the latest ground truth beacon measurements.
        """
        if self.latest_beacon_gt is None:
            return
        noisy_msg = BeaconData()
        for i in range(len(self.latest_beacon_gt.ids)):
            curr_bearing, curr_range = self.latest_beacon_gt.bearings[i], self.latest_beacon_gt.ranges[i]

            if curr_range > self.BEACON_MAX_RANGE:
                continue # if beacon is out of range,ignore it

            noisy_msg.ids.append(self.latest_beacon_gt.ids[i])
            noisy_msg.ranges.append(
                random.gauss(curr_range, self.BEACON_RANGE_NOISE + curr_range*self.BEACON_RANGE_PROP_NOISE)
            )
            noisy_msg.bearings.append(
                random.gauss(curr_bearing, self.BEACON_BEARING_NOISE)
            )
        self.beacon_noisy_pub.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NoiseInjector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
