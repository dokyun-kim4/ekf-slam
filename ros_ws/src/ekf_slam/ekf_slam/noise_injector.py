import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import random

class NoiseInjector(Node):
    def __init__(self):
        super().__init__('noise_injector')

        self.MOTOR_LINEAR_NOISE = 0.1 # m/s
        self.MOTOR_ANGULAR_NOISE = 0.05 # rad/s

        self.GPS_X_NOISE = 0.1 # m
        self.GPS_Y_NOISE = 0.1 # m

        self.BEACON_RANGE_NOISE = 0.5 # m
        self.BEACON_RANGE_PROP_NOISE = 0.05 # m
        self.BEACON_BEARING_NOISE = 0.523 # rad

        self.ENCODER_LINEAR_NOISE = 0.05 # m/s
        self.ENCODER_LINEAR_NOISE_RATIO = 0.01 # m/s
        self.ENCODER_ANGULAR_NOISE = 0.1 # rad/s
        self.ENCODER_ANGULAR_NOISE_RATIO = 0.1 # rad/s

        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.make_noisy_cmd_vel, 10)
        self.cmd_vel_noisy_pub = self.create_publisher(Twist, 'cmd_vel_noisy', 10)

        # using odom topic as gps "ground truth"
        self.gps_sub = self.create_subscription(Odometry, 'odom', self.make_noisy_gps, 10)
        self.gps_noisy_pub = self.create_publisher(Pose2D, 'gps_noisy', 10)

        self.beacon_sub = self.create_subscription(Pose2D, 'beacon', self.make_noisy_beacon, 10)
        self.beacon_noisy_pub = self.create_publisher(Pose2D, 'beacon_noisy', 10)

    def make_noisy_cmd_vel(self, msg):
        noisy_msg = Twist()
        # Our lin_vel command is in X axis since it is a differential drive robot
        noisy_msg.linear.x = msg.linear.x + random.gauss(0, self.MOTOR_LINEAR_NOISE)
        # Similarly, our ang_vel command is in Z axis
        noisy_msg.angular.z = msg.angular.z + random.gauss(0, self.MOTOR_ANGULAR_NOISE)
        self.cmd_vel_noisy_pub.publish(noisy_msg)
    
    def make_noisy_gps(self, msg):
        noisy_msg = Pose2D()
        noisy_msg.x = msg.pose.pose.position.x + random.gauss(0, self.GPS_X_NOISE)
        noisy_msg.y = msg.pose.pose.position.y + random.gauss(0, self.GPS_Y_NOISE)
        self.gps_noisy_pub.publish(noisy_msg)
    
    def make_noisy_beacon(self, msg):
        pass




def main(args=None):
    rclpy.init(args=args)
    node = NoiseInjector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
