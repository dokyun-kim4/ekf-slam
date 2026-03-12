"""
This file defines the EKFSLAMNode class, which implements the Extended Kalman Filter SLAM algorithm for a differential drive robot.
The node subscribes to control inputs from encoders, beacon measurements, and GPS data, and publishes the estimated robot pose and beacon
positions with their associated covariances. The EKF SLAM algorithm is implemented in the `ekf_predict`, `ekf_update_beacon`, and `ekf_update_gps` methods,
which handle the prediction and update steps of the filter. The node also includes dynamic time step calculation based on message timing to improve
estimation accuracy.
"""

import rclpy
from rclpy.node import Node
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Twist, Pose2D, PoseWithCovarianceStamped
from ekf_interfaces.msg import BeaconData
import numpy as np
from math import atan2, cos, sin, sqrt

from .utils.utils import wrap_angle


class EKFSLAMNode(Node):
    """
    Main node for EKF SLAM. This node will handle the EKF SLAM algorithm, including state estimation and map building.
    """

    def __init__(self):
        super().__init__("ekf_slam_node")

        # TODO: need to make this a parameter
        self.GPS_X_NOISE = 0.1  # m
        self.GPS_Y_NOISE = 0.1  # m

        self.RANGE_NOISE = 0.1
        self.RANGE_PROP_NOISE = 0.05
        self.BEARING_NOISE = 0.08

        self.num_beacons = 3

        # Run predict step whenever we have new control input
        self.create_subscription(Twist, "/encoder_noisy", self.ekf_predict, 10)
        # Run update step whenever we get observation from beacon or GPS
        self.create_subscription(
            BeaconData, "/beacon_noisy", self.ekf_update_beacon, 10
        )
        self.create_subscription(Pose2D, "/gps_noisy", self.ekf_update_gps, 10)

        # Publish the current state estimate for visualization
        self.ekf_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ekf_prediction", 10
        )
        self.ekf_beacon_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ekf_beacon_prediction", 10
        )

        
        self.last_control_time = None
        self.DT = 0.2  # Default fallback value
        # State vector keeps track of robot state [x,y,theta] and beacon info [xi, yi].
        self.x = np.zeros((3 + 2 * self.num_beacons, 1))
        # Robot state is initialized at the origin
        self.P = np.zeros((3 + 2 * self.num_beacons, 3 + 2 * self.num_beacons))
        # Initial beacon covariance is large indicating high uncertainty
        self.P[3:, 3:] = np.diag(1e3 * np.ones(2 * self.num_beacons))

        # Pass-through matrix M
        self.M = np.zeros((3, 3 + 2 * self.num_beacons))
        self.M[:3, :3] = np.eye(3)

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.

        Returns:
            Q (np.ndarray): Process noise covariance matrix
        """
        stdev_xy = 0.01
        stdev_theta = 0.05
        return np.diag([stdev_xy**2, stdev_xy**2, stdev_theta**2])

    def motion_model(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Computes the predicted state based on the differential drive motion model.

        Args:
            x (np.ndarray): Current state vector (3 + 2*num_beacons, 1)
            u (np.ndarray): Control input vector (2, 1) containing v and w
            dt (float): Time step

        Returns:
            x_t (np.ndarray): Predicted state vector
        """
        v, w = u.flatten()
        r = v / w if w != 0 else 0.0
        theta = x[2, 0]

        # Diff drive motion model
        B = np.array(
            [
                [-r * sin(theta) + r *sin(theta + w * dt)],
                [r * cos(theta) - r * cos(theta + w * dt)],
                [w * dt],
            ]
        )

        x_t = x + self.M.T @ B
        x_t[2, 0] = wrap_angle(x_t[2, 0])
        return x_t

    def motion_jacobian(self, x, u, dt):
        """
        Computes the Jacobian of the motion model with respect to the state.

        Args:
            x (np.ndarray): Current state vector (3 + 2*num_beacons, 1)
            u (np.ndarray): Control input vector (2, 1) containing v and w
            dt (float): Time step

        Returns:
            G_t (np.ndarray): Jacobian matrix of the motion model with respect to the state
        """
        v, w = u.flatten()
        r = v / w if w != 0 else 0.0
        theta = x[2, 0]

        # Hardcode jacobian expression with dynamic dt
        J1 = -r * np.cos(theta) + r * np.cos(theta + w * dt)
        J2 = -r * np.sin(theta) - r * np.sin(theta + w * dt)
        J3 = 0.0

        J = np.array([[0.0, 0.0, J1], [0.0, 0.0, J2], [0.0, 0.0, J3]], dtype=np.float64)

        # G = I + M.T @ J @ M
        G_t = np.eye(3 + 2 * self.num_beacons) + self.M.T @ J @ self.M
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
                self.get_logger().warn(
                    f"Invalid dt value: {dt} seconds, using default {self.DT}s"
                )
                dt = self.DT

        self.last_control_time = current_time
        self.get_logger().debug(f"EKF predict dt: {dt:.4f} seconds")

        # control input `u`
        u = np.array([[msg.linear.x], [msg.angular.z]])
        # Compute the motion jacobian G with dynamic dt
        G_t = self.motion_jacobian(self.x, u, dt)

        # State prediction with dynamic dt
        self.x = self.motion_model(self.x, u, dt)
        self.x[2, 0] = wrap_angle(self.x[2, 0])
        # Covariance prediction
        self.P = G_t @ self.P @ G_t.T + self.M.T @ self.get_Q() @ self.M

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.pose.position.x = float(self.x[0, 0])
        pose_msg.pose.pose.position.y = float(self.x[1, 0])

        # need to convert back to quaternion
        q = quaternion_from_euler(0, 0, float(self.x[2, 0]))
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]

        # self.P only has x,y theta but we need a 6 x 6
        P_66 = np.zeros((6, 6))
        P_66[0:2, 0:2] = self.P[0:2, 0:2]  # xy covariance
        P_66[5, 5] = self.P[2, 2]  # theta variance
        P_66[5, 0:2] = self.P[2, 0:2]  # theta-xy covariance
        P_66[0:2, 5] = self.P[0:2, 2]  # theta-xy covariance

        pose_msg.pose.covariance = P_66.flatten()
        self.ekf_pub.publish(pose_msg)

    def ekf_update_beacon(self, msg: BeaconData):
        """
        EKF update step triggered by new beacon measurements.
        """

        # Note that our z vector has [range, bearing, id] in this order
        # need to sort by ID since order of measurements may change
        all_measurements = [
            np.array([r, b, id])[:, None]
            for id, r, b in sorted(zip(msg.ids, msg.ranges, msg.bearings))
        ]

        # We need to iterate through the list of beacon measurements,
        # and update state using each measurement.
        for measurement in all_measurements:
            z_t = measurement[0:2]  # Extract range and bearing
            id = int(measurement[2, 0])  # Extract ID

            x_t, y_t = self.x[0, 0], self.x[1, 0]
            beacon_x_idx, beacon_y_idx = 3 + 2 * (id - 1), 4 + 2 * (id - 1)

            # New beacon will have its state initialized based on the first measurement.
            # Note both robot state and all beacons start at (0,0), and if we try to compute the value of
            # (m_x - x_t) ** 2 + (m_y - y_t) ** 2, it will result in 0, which causes divide by zero errors in the Jacobian and Kalman Gain calculations.
            if self.x[beacon_x_idx] == 0 and self.x[beacon_y_idx] == 0:
                self.get_logger().info("New beacon, initializing")
                r, b = z_t[0, 0], z_t[1, 0]
                robot_theta = self.x[2, 0]
                self.x[beacon_x_idx, 0] = x_t + r * cos(b + robot_theta)
                self.x[beacon_y_idx, 0] = y_t + r * sin(b + robot_theta)

            m_x, m_y = self.x[beacon_x_idx, 0], self.x[beacon_y_idx, 0]

            # Compute sensor noise R, Note it is 2x2 since we ignore ID
            range_stdev = self.RANGE_NOISE + z_t[0, 0] * self.RANGE_PROP_NOISE
            bearing_stdev = self.BEARING_NOISE
            R = np.diag([range_stdev, bearing_stdev]) ** 2

            # First, linearize the measurement function
            q_t = (m_x - x_t) ** 2 + (m_y - y_t) ** 2
            sq_t = sqrt(q_t)
            delta_x = x_t - m_x
            delta_y = y_t - m_y

            h_t = np.array(
                [
                    [
                        delta_x / sq_t,
                        delta_y / sq_t,
                        0.0,
                        -delta_x / sq_t,
                        -delta_y / sq_t,
                    ],
                    [
                        -delta_y / q_t,
                        delta_x / q_t,
                        -1.0,
                        delta_y / q_t,
                        -delta_x / q_t,
                    ],
                ]
            )

            # Passthru matrix for current beacon
            F = np.zeros((5, 3 + 2 * self.num_beacons))
            # Robot pose part of the Jacobian
            F[0:3, 0:3] = np.eye(3)

            # Beacon pose part of Jacobian
            F[3:5, 3 + 2 * (id - 1) : 3 + 2 * (id)] = np.eye(2)

            # Measurement Jacobian
            H_t = h_t @ F

            # Compute innovation and Kalman Gain
            S_t = H_t @ self.P @ H_t.T + R
            K_t = self.P @ H_t.T @ np.linalg.inv(S_t)

            # FIX 2: Wrap the difference/innovation, not just the expectation itself
            y_t = z_t - np.array(
                [
                    [sqrt(q_t)],
                    [atan2(-delta_y, -delta_x) - self.x[2, 0]],
                ]
            )
            y_t[1, 0] = wrap_angle(y_t[1, 0])

            # Update State and Covariance
            self.x += K_t @ y_t
            self.x[2, 0] = wrap_angle(self.x[2, 0])
            self.P = (np.eye(3 + 2 * self.num_beacons) - K_t @ H_t) @ self.P

        current_time = self.get_clock().now()
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.pose.position.x = float(self.x[0, 0])
        pose_msg.pose.pose.position.y = float(self.x[1, 0])

        # need to convert back to quaternion
        q = quaternion_from_euler(0, 0, float(self.x[2, 0]))
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]

        # self.P only has x,y theta but we need a 6 x 6
        P_66 = np.zeros((6, 6))
        P_66[0:2, 0:2] = self.P[0:2, 0:2]  # xy covariance
        P_66[5, 5] = self.P[2, 2]  # theta variance
        P_66[5, 0:2] = self.P[2, 0:2]  # theta-xy covariance
        P_66[0:2, 5] = self.P[0:2, 2]  # theta-xy covariance

        pose_msg.pose.covariance = P_66.flatten()
        self.ekf_pub.publish(pose_msg)

        # Also need to publish covariance for beacon pose estimate
        for idx in range(self.num_beacons):
            beacon_pose_msg = PoseWithCovarianceStamped()
            beacon_pose_msg.header.stamp = current_time.to_msg()
            beacon_pose_msg.header.frame_id = f"beacon_{idx + 1}"
            beacon_pose_msg.pose.pose.position.x = float(self.x[3 + 2 * idx, 0])
            beacon_pose_msg.pose.pose.position.y = float(self.x[4 + 2 * idx, 0])
            beacon_pose_msg.pose.pose.position.z = 0.5

            # Note we only have x y covariance
            P_66 = np.zeros((6, 6))
            P_66[0:2, 0:2] = self.P[
                3 + 2 * idx : 5 + 2 * idx, 3 + 2 * idx : 5 + 2 * idx
            ]  # beacon xy covariance
            beacon_pose_msg.pose.covariance = P_66.flatten()
            self.ekf_beacon_pub.publish(beacon_pose_msg)

    def ekf_update_gps(self, msg: Pose2D):
        """
        EKF update step triggered by new GPS measurements.
        """
        z = np.array([[msg.x], [msg.y]])
        H = np.zeros((2, 3 + 2 * self.num_beacons))
        H[0:2, 0:2] = np.eye(2)

        R = np.diag([self.GPS_X_NOISE**2, self.GPS_Y_NOISE**2])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2, 0] = wrap_angle(self.x[2, 0])
        self.P = (np.eye(3 + 2 * self.num_beacons) - K @ H) @ self.P

        current_time = self.get_clock().now()
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.pose.position.x = float(self.x[0, 0])
        pose_msg.pose.pose.position.y = float(self.x[1, 0])

        # need to convert back to quaternion
        q = quaternion_from_euler(0, 0, float(self.x[2, 0]))
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]

        # self.P only has x,y theta but we need a 6 x 6
        P_66 = np.zeros((6, 6))
        P_66[0:2, 0:2] = self.P[0:2, 0:2]  # xy covariance
        P_66[5, 5] = self.P[2, 2]  # theta variance
        P_66[5, 0:2] = self.P[2, 0:2]  # theta-xy covariance
        P_66[0:2, 5] = self.P[0:2, 2]  # theta-xy covariance

        pose_msg.pose.covariance = P_66.flatten()
        self.ekf_pub.publish(pose_msg)


def main(args=None):
    """
    Entry point for the EKF SLAM node.
    """
    rclpy.init(args=args)
    ekf_slam_node = EKFSLAMNode()
    rclpy.spin(ekf_slam_node)
    ekf_slam_node.destroy_node()
    rclpy.shutdown()
