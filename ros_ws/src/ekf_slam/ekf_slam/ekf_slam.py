import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from ekf_interfaces.msg import BeaconData
import numpy as np

class EKFSLAMNode(Node):
    """
    Main node for EKF SLAM. This node will handle the EKF SLAM algorithm, including state estimation and map building.
    """
    def __init__(self):
        super().__init__('ekf_slam_node')

        self.GPS_X_NOISE = 0.1 # m
        self.GPS_Y_NOISE = 0.1 # m

        # Run predict step whenever we have new control input
        self.create_subscription(Twist, '/encoder_vel', self.ekf_predict, 10)

        # Run update step whenever we get observation from beacon or GPS
        self.create_subscription(BeaconData, '/beacon_noisy', self.ekf_update_beacon, 10)
        self.create_subscription(Pose2D, '/gps_noisy', self.ekf_update_gps, 10)

        # Publish the current state estimate for visualization
        self.ekf_pub = self.create_publisher(Pose2D, '/ekf_prediction', 10)

        self.num_beacons = 3
        # Dynamic dt calculation - will be computed from message timing
        self.last_control_time = None
        self.DT = 0.2  # Default fallback value
        # State vector keeps track of robot state [x,y,theta] and beacon info [xi, yi, si] where si is the signature of the beacon
        self.x = np.zeros((3 + 3 * self.num_beacons, 1))
        # Robot state is initialized at the origin
        self.P = np.zeros((3 + 3 * self.num_beacons, 3 + 3 * self.num_beacons))
        # Initial beacon covariance is large indicating high uncertainty
        self.P[3:, 3:] = np.diag(1e3 * np.ones(3*self.num_beacons))

        # Pass-through matrix M
        self.M = np.zeros((3, 3 + 3 * self.num_beacons))
        self.M[:3, :3] = np.eye(3)
    
    def wrap_angle(self, angle):
        """
        Wraps the given angle to the range [-pi, pi].

        Args:
            angle (float): The angle in radians
        
        Returns:
            float: The wrapped angle in radians
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.

        Returns:
            Q (np.ndarray): Process noise covariance matrix
        """
        stdev = 0.01
        return np.diag([stdev**2, stdev**2, stdev**2])

    def motion_model(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Computes the predicted state based on the differential drive motion model.

        Args:
            x (np.ndarray): Current state vector (3 + 3*num_beacons, 1)
            u (np.ndarray): Control input vector (2, 1) containing v and w
            dt (float): Time step

        Returns:
            x_t (np.ndarray): Predicted state vector
        """
        v,w = u.flatten()
        r = v / w if w != 0 else 0.0
        theta = x[2,0]

        # Diff drive motion model with dynamic dt
        B = np.array([[-r*np.sin(theta) + r*np.sin(theta + w*dt)],
                      [r*np.cos(theta) - r*np.cos(theta + w*dt)],
                      [w*dt]])

        x_t = x + self.M.T @ B
        x_t[2,0] = self.wrap_angle(x_t[2,0])
        return x_t

    def motion_jacobian(self, x, u, dt):
        """
        Computes the Jacobian of the motion model with respect to the state.

        Args:
            x (np.ndarray): Current state vector (3 + 3*num_beacons, 1)
            u (np.ndarray): Control input vector (2, 1) containing v and w
            dt (float): Time step
        
        Returns:
            G_t (np.ndarray): Jacobian matrix of the motion model with respect to the state
        """
        v,w = u.flatten()
        r = v / w if w != 0 else 0.0
        theta = x[2,0]

        # Hardcode jacobian expression with dynamic dt
        J1 = -r*np.cos(theta) + r*np.cos(theta + w*dt)
        J2 = -r*np.sin(theta) - r*np.sin(theta + w*dt)
        J3 = 0.0

        J = np.array([[0.0, 0.0, J1],
                      [0.0, 0.0, J2],
                      [0.0, 0.0, J3]], dtype=np.float64)
        
        # G = I + M.T @ J @ M
        G_t = np.eye(3 + 3 * self.num_beacons) + self.M.T @ J @ self.M
        return G_t

    def ekf_predict(self, msg: Twist):
        """
        EKF prediction step using control inputs.

        Args:
            msg (Twist): Control input message containing v and w
        """
        current_time = self.get_clock().now()
        
        # Calculate dt from message timing
        if self.last_control_time is None:
            # First message - use default dt
            dt = self.DT
            self.get_logger().info(f"EKF SLAM initialized with default dt: {dt}")
        else:
            duration = current_time - self.last_control_time
            dt = duration.nanoseconds / 1e9
            
            # Guard against unreasonable dt values
            if dt <= 0 or dt > 0.5:  # If dt is negative or too large
                self.get_logger().warn(f"Invalid dt value: {dt} seconds, using default {self.DT}s")
                dt = self.DT
            
        self.last_control_time = current_time
        self.get_logger().debug(f"EKF predict dt: {dt:.4f} seconds")
        
        # control input `u`
        u = np.array([[msg.linear.x], [msg.angular.z]])
        # Compute the motion jacobian G with dynamic dt
        G_t = self.motion_jacobian(self.x, u, dt)

        # State prediction with dynamic dt
        self.x = self.motion_model(self.x, u, dt)
        self.x[2, 0] = self.wrap_angle(self.x[2, 0])
        # Covariance prediction
        self.P = G_t @ self.P @ G_t.T + self.M.T @ self.get_Q() @ self.M

        pose_msg = Pose2D()
        pose_msg.x = float(self.x[0, 0])
        pose_msg.y = float(self.x[1, 0])
        pose_msg.theta = float(self.x[2, 0])
        self.ekf_pub.publish(pose_msg)
    
    def ekf_update_beacon(self, msg: BeaconData):
        """
        EKF update step triggered by new beacon measurements.
        """
        pass

    def ekf_update_gps(self, msg: Pose2D):
        """
        EKF update step triggered by new GPS measurements.
        """
        pass

def main(args=None):
    """
    Entry point for the EKF SLAM node.
    """
    rclpy.init(args=args)
    ekf_slam_node = EKFSLAMNode()
    rclpy.spin(ekf_slam_node)
    ekf_slam_node.destroy_node()
    rclpy.shutdown()
