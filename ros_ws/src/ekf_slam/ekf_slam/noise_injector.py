import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
import random
from typing import Optional, cast


class NoiseInjector(Node):
    def __init__(self):
        super().__init__(
            "noise_injector",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # --- Sensor Parameters --- #

        # cmd_vel only publishes once until there is a new command,
        # so we need to inject noise at a fixed interval to simulate varying motor noise.
        self.MOTOR_LINEAR_NOISE = cast(
            float,
            self.get_parameter_or(
                "motor_linear_noise",
                Parameter("motor_linear_noise", Parameter.Type.DOUBLE, 0.1),
            ).value,
        )  # m/s
        self.MOTOR_ANGULAR_NOISE = cast(
            float,
            self.get_parameter_or(
                "motor_angular_noise",
                Parameter("motor_angular_noise", Parameter.Type.DOUBLE, 0.05),
            ).value,
        )  # rad/s
        self.MOTOR_STEP_INTERVAL = cast(
            float,
            self.get_parameter_or(
                "motor_step_interval",
                Parameter("motor_step_interval", Parameter.Type.DOUBLE, 0.2),
            ).value,
        )  # seconds

        # Encoder interval should match motor step interval
        self.ENCODER_LINEAR_NOISE = cast(
            float,
            self.get_parameter_or(
                "encoder_linear_noise",
                Parameter("encoder_linear_noise", Parameter.Type.DOUBLE, 0.05),
            ).value,
        )  # m/s
        self.ENCODER_LINEAR_NOISE_RATIO = cast(
            float,
            self.get_parameter_or(
                "encoder_linear_noise_ratio",
                Parameter("encoder_linear_noise_ratio", Parameter.Type.DOUBLE, 0.01),
            ).value,
        )  # m/s
        self.ENCODER_ANGULAR_NOISE = cast(
            float,
            self.get_parameter_or(
                "encoder_angular_noise",
                Parameter("encoder_angular_noise", Parameter.Type.DOUBLE, 0.1),
            ).value,
        )  # rad/s
        self.ENCODER_ANGULAR_NOISE_RATIO = cast(
            float,
            self.get_parameter_or(
                "encoder_angular_noise_ratio",
                Parameter("encoder_angular_noise_ratio", Parameter.Type.DOUBLE, 0.1),
            ).value,
        )  # rad/s
        self.ENCODER_INTERVAL = cast(
            float,
            self.get_parameter_or(
                "encoder_interval",
                Parameter("encoder_interval", Parameter.Type.DOUBLE, 0.2),
            ).value,
        )  # seconds

        self.GPS_X_NOISE = cast(
            float,
            self.get_parameter_or(
                "gps_x_noise", Parameter("gps_x_noise", Parameter.Type.DOUBLE, 0.1)
            ).value,
        )  # m
        self.GPS_Y_NOISE = cast(
            float,
            self.get_parameter_or(
                "gps_y_noise", Parameter("gps_y_noise", Parameter.Type.DOUBLE, 0.1)
            ).value,
        )  # m
        self.GPS_INTERVAL = cast(
            float,
            self.get_parameter_or(
                "gps_interval", Parameter("gps_interval", Parameter.Type.DOUBLE, 1.0)
            ).value,
        )  # seconds

        # --- Stored Ground Truth State --- #
        self.latest_cmd_vel_gt: Optional[Twist] = None
        self.latest_cmd_vel_actual: Optional[Twist] = None  # need this for encoder
        self.latest_gps_gt: Optional[Odometry] = None

        # --- Subscribers, Publishers, and Timers --- #
        self.motor_step_timer = self.create_timer(
            self.MOTOR_STEP_INTERVAL, self.make_noisy_cmd_vel
        )
        self.gps_timer = self.create_timer(self.GPS_INTERVAL, self.make_noisy_gps)
        self.encoder_timer = self.create_timer(
            self.ENCODER_INTERVAL, self.make_noisy_encoder
        )

        # These subscribers just store the latest ground truth data
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.gps_sub = self.create_subscription(
            Odometry, "/ground_truth", self.gps_callback, 10
        )

        # Publish noisy data
        self.cmd_vel_noisy_pub = self.create_publisher(Twist, "/cmd_vel_noisy", 10)
        self.gps_noisy_pub = self.create_publisher(Pose2D, "/gps_noisy", 10)
        self.encoder_vel_pub = self.create_publisher(Twist, "/encoder_noisy", 10)

    # --- Simple Timer Callbacks to update states --- #
    def gps_callback(self, msg: Odometry):
        """
        Updates the latest ground truth GPS message

        Args:
            msg (Odometry): The incoming ground truth GPS odometry message
        """
        self.latest_gps_gt = msg

    def cmd_vel_callback(self, msg: Twist):
        """
        Updates the latest ground truth velocity command

        Args:
            msg (Twist): The incoming velocity command message
        """
        self.latest_cmd_vel_gt = msg

    # --- Noise injection for spoofed sensor data--- #

    def make_noisy_cmd_vel(self):
        """
        Simulates and publishes a noisy velocity command based on the latest ground truth

        Args:
            None: No arguments

        Returns:
            None: Publish to `/cmd_vel_noisy`
        """
        if self.latest_cmd_vel_gt is None:
            return
        noisy_cmd = Twist()
        # Our lin_vel command is in X axis since it is a differential drive robot
        noisy_cmd.linear.x = self.latest_cmd_vel_gt.linear.x + random.gauss(
            0, self.MOTOR_LINEAR_NOISE
        )
        # Similarly, our ang_vel command is in Z axis
        noisy_cmd.angular.z = self.latest_cmd_vel_gt.angular.z + random.gauss(
            0, self.MOTOR_ANGULAR_NOISE
        )
        self.latest_cmd_vel_actual = noisy_cmd
        self.cmd_vel_noisy_pub.publish(noisy_cmd)

    def make_noisy_gps(self):
        """
        Simulates and publishes a noisy GPS reading based on the latest ground truth GPS position

        Args:
            None: No arguments

        Returns:
            None: Publishes to `/gps_noisy`
        """
        if self.latest_gps_gt is None:
            return
        noisy_msg = Pose2D()
        noisy_msg.x = self.latest_gps_gt.pose.pose.position.x + random.gauss(
            0, self.GPS_X_NOISE
        )
        noisy_msg.y = self.latest_gps_gt.pose.pose.position.y + random.gauss(
            0, self.GPS_Y_NOISE
        )
        self.gps_noisy_pub.publish(noisy_msg)

    def make_noisy_encoder(self):
        """
        Simulates and publishes noisy encoder readings based on the latest actual cmd_vel

        Args:
            None: No arguments

        Returns:
            None: Publishes to `/encoder_noisy`
        """
        if self.latest_cmd_vel_actual is None:
            return
        encoder_vel = Twist()
        encoder_vel.linear.x = random.gauss(
            self.latest_cmd_vel_actual.linear.x,
            self.ENCODER_LINEAR_NOISE
            + abs(self.latest_cmd_vel_actual.linear.x)
            * self.ENCODER_LINEAR_NOISE_RATIO,
        )
        encoder_vel.angular.z = random.gauss(
            self.latest_cmd_vel_actual.angular.z,
            self.ENCODER_ANGULAR_NOISE
            + abs(self.latest_cmd_vel_actual.angular.z)
            * self.ENCODER_ANGULAR_NOISE_RATIO,
        )
        self.encoder_vel_pub.publish(encoder_vel)


def main(args=None):
    """
    Spin up the node
    """
    rclpy.init(args=args)
    node = NoiseInjector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
