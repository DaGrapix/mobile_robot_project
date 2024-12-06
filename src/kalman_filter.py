import numpy as np
from typing import Optional


class KalmanFilter():
    def __init__(self,
                 robot_pos_m: np.ndarray,
                 robot_angle_m: float,
                 speed_convertor: float,
                 DIST_WHEEL: float,
                 CAM_POS_MEASUREMENT_VAR,
                 CAM_ANGLE_MEASUREMENT_VAR,
                 PROCESS_POS_VAR,
                 PROCESS_ANGLE_VAR,
                 ):
        """Kalman filter class"""

        self.A = np.eye(3)  # State transition matrix
        self.H = np.eye(3)  # Observation matrix
        self.R = np.diag([CAM_POS_MEASUREMENT_VAR, CAM_POS_MEASUREMENT_VAR, CAM_ANGLE_MEASUREMENT_VAR])  # Observation noise covariance
        self.Q = np.diag([PROCESS_POS_VAR, PROCESS_POS_VAR, PROCESS_ANGLE_VAR]) # Process noise covariance
        self.DIST_WHEEL = DIST_WHEEL  # Distance between wheels

        # initialize the state and covariance
        self.x_k1 = np.array([robot_pos_m[0], robot_pos_m[1], robot_angle_m])
        self.P_k1 = np.eye(3)

        self.speed_convertor = speed_convertor

    def B_matrix(self, theta: float, dt: float) -> np.ndarray:
        """Measurement matrix"""
        B = dt * np.array([[0.5*np.cos(theta), 0.5*np.cos(theta)],
                           [0.5*np.sin(theta), 0.5*np.sin(theta)],
                           [-1/self.DIST_WHEEL, 1/self.DIST_WHEEL]])
        return B

    def predict(self, u: np.ndarray, dt) -> None:
        """Predict the next state"""
        B = self.B_matrix(self.x_k1[2], dt)
        x_k = self.A @ self.x_k1 + B @ u
        P_k = self.A @ self.P_k1 @ self.A.T + self.Q

        self.x_k1 = x_k
        self.P_k1 = P_k

    def update(self, z: np.ndarray) -> None:
        """Update the state"""
        innovation = z - self.H @ self.x_k1
        self.S = self.R + self.H @ self.P_k1 @ self.H.T
        self.K = self.P_k1 @ self.H.T @ np.linalg.inv(self.S)

        self.x_k1 = self.x_k1 + self.K @ innovation
        self.P_k1 = (np.eye(3) - self.K @ self.H) @ self.P_k1

    def estimate(self, u: np.ndarray, z: Optional[np.ndarray], dt) -> np.ndarray:
        """State transition function"""
        u = self.speed_convertor(u)
        self.predict(u, dt)

        if z is not None:
            self.update(z)

        return self.x_k1[:2], self.x_k1[2], self.P_k1
