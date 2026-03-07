"""
This node is responsible for visualizing the different paths of the EKF SLAM output.
- Ground truth path of the robot
- Dead reckoning path of the robot
- GPS path of the robot
- EKF SLAM estimated path of the robot
"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
import tf_transformations
import numpy as np

class PathVizNode(Node):
    
    def __init__(self):
        super().__init__('path_viz_node')
        
        # Create publishers for the different paths
        self.gt_path_pub = self.create_publisher(Path, '/ground_truth_path', 10)
        self.dr_path_pub = self.create_publisher(Path, '/dead_reckoning_path', 10)
        self.gps_path_pub = self.create_publisher(Path, '/gps_path', 10)
        self.ekf_path_pub = self.create_publisher(Path, '/ekf_slam_path', 10)
        
        # Create subscribers for the different topics to use for path generation
        self.create_subscription(Odometry, '/ground_truth', self.plot_gt, 10)
        self.create_subscription(Twist, '/cmd_vel_noisy', self.plot_dead_reckoning, 10)
        self.create_subscription(Pose2D, '/gps_noisy', self.plot_gps, 10)
        self.create_subscription(Path, '/predicted_pose', self.plot_ekf, 10)

        self.latest_gt_pose = None
        self.gt_path = Path()
        self.dr_path = Path()
        self.gps_path = Path()
        self.ekf_path = Path()

    def plot_gt(self, msg: Odometry):
        """
        Plot the ground truth path of the robot as a Path message in RViz2.
        TheOdometry message from the p3d plugin is converted to a PoseStamped message
        that is appended to the Path message.

        Args:
            msg (Odometry): Odometry message containing the ground truth pose of the robot.
        """
        # Pocket the latest ground truth pose for use in dead reckoning
        self.latest_gt_pose = msg
        
        pose_msg = PoseStamped()
        # Convert Odometry message to Pose 
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_footprint'
        pose_msg.pose.position.x = msg.pose.pose.position.x
        pose_msg.pose.position.y = msg.pose.pose.position.y

        quat = [msg.pose.pose.orientation.x, 
                msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, 
                msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        pose_msg.pose.orientation.z = yaw

        # Set the frame_id for the Path message
        self.gt_path.header.frame_id = 'base_footprint'
        self.gt_path.header.stamp = self.get_clock().now().to_msg()
        self.gt_path.poses.append(pose_msg) # type: ignore
        self.gt_path_pub.publish(self.gt_path)

    def plot_dead_reckoning(self, msg: Twist):
        """
        Plot the dead reckoning path of the robot as a Path message in RViz2.
        The Twist message from the `cmd_vel_noisy` topic is converted to a PoseStamped message
        using a differential drive kinematic model and appended to the Path message.

        Args:
            msg (Twist): Twist message containing the velocity commands for the robot.
        """
        if self.latest_gt_pose is None:
            self.get_logger().warn("No ground truth pose received yet, cannot plot dead reckoning path.")
            return
        v, w = msg.linear.x, msg.angular.z
        

        x = self.latest_gt_pose.pose.pose.position.x
        y = self.latest_gt_pose.pose.pose.position.y
        quat = [self.latest_gt_pose.pose.pose.orientation.x, 
                self.latest_gt_pose.pose.pose.orientation.y, 
                self.latest_gt_pose.pose.pose.orientation.z, 
                self.latest_gt_pose.pose.pose.orientation.w]
        _, _, theta = tf_transformations.euler_from_quaternion(quat)

        # Get the current time from the node's clock
        current_time = self.get_clock().now()
        latest_gt_time = Time.from_msg(self.latest_gt_pose.header.stamp)
        duration = current_time - latest_gt_time
        dt = duration.nanoseconds / 1e9
        
        # Get deltas with diff drive model
        dx = np.cos(theta) * v * dt
        dy = np.sin(theta) * v *dt
        dtheta = w * dt

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_footprint'
        pose_msg.pose.position.x = x+dx
        pose_msg.pose.position.y = y+dy
        pose_msg.pose.orientation.z = theta+dtheta

        self.dr_path.header.frame_id = 'base_footprint'
        self.dr_path.header.stamp = self.get_clock().now().to_msg()
        self.dr_path.poses.append(pose_msg) # type: ignore
        self.dr_path_pub.publish(self.dr_path)

    def plot_gps(self, msg: Pose2D):
        """
        Plot the GPS path of the robot as a Path message in RViz2.
        The Pose2D message from the `gps` topic is converted to a PoseStamped message
        and appended to the Path message.

        Note that GPS only provides x and y positions of the robot
        
        Args:
            msg (Pose2D): Pose2D message containing the GPS pose of the robot.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_footprint'
        pose_msg.pose.position.x = msg.x
        pose_msg.pose.position.y = msg.y

        # Set the frame_id for the Path message
        self.gps_path.header.frame_id = 'base_footprint'
        self.gps_path.header.stamp = self.get_clock().now().to_msg()
        self.gps_path.poses.append(pose_msg) # type: ignore
        self.gps_path_pub.publish(self.gps_path)

    def plot_ekf(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PathVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
