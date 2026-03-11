"""
This node is responsible for visualizing the different paths of the EKF SLAM output.
- Ground truth path of the robot
- Dead reckoning path of the robot
- GPS path of the robot
- EKF SLAM estimated path of the robot
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose2D, PoseStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from ekf_interfaces.msg import BeaconData
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
        self.beacon_viz_pub = self.create_publisher(MarkerArray, '/beacon_viz', 10)
        
        # Create subscribers for the different topics to use for path generation
        self.create_subscription(Odometry, '/ground_truth', self.plot_gt, 10)
        self.create_subscription(Twist, '/encoder_noisy', self.plot_dead_reckoning, 10)
        self.create_subscription(Pose2D, '/gps_noisy', self.plot_gps, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/ekf_prediction', self.plot_ekf, 10)
        self.create_subscription(BeaconData, '/beacon_ground_truth', self.plot_beacons, 10)

        self.latest_gt_pose = None
        self.last_twist_time = None
        self.last_twist_pose = None  # Track last dead reckoning pose
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
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = msg.pose.pose.position.x
        pose_msg.pose.position.y = msg.pose.pose.position.y

        quat = [msg.pose.pose.orientation.x, 
                msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, 
                msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        pose_msg.pose.orientation.z = yaw

        # Set the frame_id for the Path message
        self.gt_path.header.frame_id = 'odom'
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
        current_time = self.get_clock().now()
        
        # Initialize tracking variables on first message
        if self.last_twist_time is None:
            # Initialize with ground truth pose if available, otherwise origin
            if self.latest_gt_pose is not None:
                x = self.latest_gt_pose.pose.pose.position.x
                y = self.latest_gt_pose.pose.pose.position.y
                quat = [self.latest_gt_pose.pose.pose.orientation.x, 
                        self.latest_gt_pose.pose.pose.orientation.y, 
                        self.latest_gt_pose.pose.pose.orientation.z, 
                        self.latest_gt_pose.pose.pose.orientation.w]
                _, _, theta = tf_transformations.euler_from_quaternion(quat)
            else:
                x, y, theta = 0.0, 0.0, 0.0
                
            self.last_twist_pose = [x, y, theta]
            self.last_twist_time = current_time
            self.get_logger().info(f"Initializing dead reckoning at ({x:.3f}, {y:.3f}, {theta:.3f})")
            return
        
        # Calculate dt from message timing
        duration = current_time - self.last_twist_time
        dt = duration.nanoseconds / 1e9
        
        # Guard against unreasonable dt values
        if dt <= 0 or dt > 0.5:  # If dt is negative or too large (> 0.5 second)
            self.get_logger().warn(f"Invalid dt value: {dt} seconds, skipping update")
            return
            
        v, w = msg.linear.x, msg.angular.z
        x, y, theta = self.last_twist_pose # type: ignore
        
        # Use exact analytical differential drive model
        if abs(w) < 1e-6:  # Nearly straight line motion
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0.0
        else:
            r = v / w
            dx = -r * np.sin(theta) + r * np.sin(theta + w * dt)
            dy = r * np.cos(theta) - r * np.cos(theta + w * dt)
            dtheta = w * dt
        
        # Update pose
        new_x = x + dx
        new_y = y + dy  
        new_theta = theta + dtheta
        
        # Wrap angle to [-pi, pi] to match typical robot conventions
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
            
        # Store updated pose and time for next iteration        
        self.last_twist_pose = [new_x, new_y, new_theta]        
        self.last_twist_time = current_time

        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = new_x
        pose_msg.pose.position.y = new_y
        pose_msg.pose.orientation.z = new_theta

        self.dr_path.header.frame_id = 'odom'
        self.dr_path.header.stamp = current_time.to_msg()
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
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = msg.x
        pose_msg.pose.position.y = msg.y

        # Set the frame_id for the Path message
        self.gps_path.header.frame_id = 'odom'
        self.gps_path.header.stamp = self.get_clock().now().to_msg()
        self.gps_path.poses.append(pose_msg) # type: ignore
        self.gps_path_pub.publish(self.gps_path)

    def plot_ekf(self, msg: PoseWithCovarianceStamped):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = msg.pose.pose.position.x
        pose_msg.pose.position.y = msg.pose.pose.position.y

        # need to convert orientation to euler angle
        _, _, pose_msg.pose.orientation.z = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        # Set the frame_id for the Path message
        self.ekf_path.header.frame_id = 'odom'
        self.ekf_path.header.stamp = self.get_clock().now().to_msg()
        self.ekf_path.poses.append(pose_msg) # type: ignore
        self.ekf_path_pub.publish(self.ekf_path)

    def plot_beacons(self, msg: BeaconData):
        """
        Plot the beacons as cylinder markers in RViz.
        """
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        for i, beacon_id in enumerate(msg.ids):
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = stamp
            marker.ns = 'beacons'
            marker.id = beacon_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(msg.x_poses[i])
            marker.pose.position.y = float(msg.y_poses[i])
            marker.pose.position.z = 0.5
            
            # Default orientation
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 1.0
            
            # Make the beacons visible with a distinct color (e.g., bright orange/yellow)
            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker) # type: ignore
            
        self.beacon_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PathVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
