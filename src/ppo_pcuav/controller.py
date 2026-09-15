import numpy as np

from .config import DroneConfig


class PID:
    def __init__(self, kp: float, ki: float, kd: float, limit: float, integral_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral_limit = integral_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_update = True

    def update(self, error: float, dt: float) -> float:
        if self.first_update:
            self.previous_error = error
            self.first_update = False
        self.integral = float(
            np.clip(
                self.integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.limit, self.limit))


class CascadedPIDController:
    """Position -> velocity -> attitude -> angular-rate cascaded controller."""

    def __init__(self, config: DroneConfig):
        self.config = config
        self.z_position = PID(3.0, 0.0, 0.0, 2.5, 0.0)
        self.z_velocity = PID(3.0, 1.0, 0.5, 5.0, 1.0)
        self.x_position = PID(2.5, 0.0, 0.0, 4.5, 0.0)
        self.x_velocity = PID(2.0, 0.1, 0.8, 2.1, 1.0)
        self.pitch = PID(1.5, 0.0, 0.0, 1.0, 0.0)
        self.pitch_rate = PID(40.0, 5.0, 1.5, config.max_torque, 10.0)

    def reset(self) -> None:
        for controller in (
            self.z_position,
            self.z_velocity,
            self.x_position,
            self.x_velocity,
            self.pitch,
            self.pitch_rate,
        ):
            controller.reset()

    def compute_control(
        self,
        state: np.ndarray,
        target_position: np.ndarray,
        target_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        x, z, theta, _, vx, vz, omega, _ = state
        target_x, target_z = target_position
        target_vx, target_vz = target_velocity
        dt = 0.02

        desired_vz = self.z_position.update(target_z - z, dt) + target_vz
        desired_az = self.z_velocity.update(desired_vz - vz, dt)
        vertical_thrust = self.config.total_mass * (
            self.config.gravity + desired_az
        )
        thrust = np.clip(
            vertical_thrust / max(np.cos(theta), 0.1),
            0.0,
            self.config.max_thrust,
        )

        desired_vx = self.x_position.update(target_x - x, dt) + target_vx
        desired_ax = self.x_velocity.update(desired_vx - vx, dt)
        desired_theta = np.clip(
            desired_ax / self.config.gravity,
            -np.radians(15.0),
            np.radians(15.0),
        )
        desired_omega = self.pitch.update(desired_theta - theta, dt)
        torque = self.pitch_rate.update(desired_omega - omega, dt)
        return np.array([thrust, torque], dtype=np.float64)
