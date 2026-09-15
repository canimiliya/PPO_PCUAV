import numpy as np

from .config import DroneConfig


def _solve_3x3(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Solve a 3x3 system with partial pivoting.

    The explicit solver avoids platform-specific BLAS crashes observed in some
    Windows NumPy installations while remaining sufficient for this small system.
    """

    augmented = np.column_stack(
        [np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64)]
    )
    for column in range(3):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column, column:] /= augmented[column, column]
        for row in range(column + 1, 3):
            augmented[row, column:] -= (
                augmented[row, column] * augmented[column, column:]
            )

    solution = np.zeros(3, dtype=np.float64)
    for row in range(2, -1, -1):
        solution[row] = augmented[row, 3] - np.dot(
            augmented[row, row + 1 : 3], solution[row + 1 : 3]
        )
    return solution


class PayloadUAVDynamics:
    """Two-dimensional nonlinear UAV-payload dynamics with RK4 integration."""

    def __init__(self, config: DroneConfig, dt: float = 0.02):
        self.config = config
        self.dt = float(dt)
        self._mass_rope = config.payload_mass * config.effective_rope_length
        self._gravity_moment = self._mass_rope * config.gravity
        self.state = np.zeros(8, dtype=np.float64)

    def reset(self, initial_state: np.ndarray | list[float] | None = None) -> np.ndarray:
        self.state = (
            np.zeros(8, dtype=np.float64)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).copy()
        )
        return self.state.copy()

    def step(self, control: np.ndarray) -> np.ndarray:
        state = self.state
        dt = self.dt
        k1 = self.derivatives(state, control)
        k2 = self.derivatives(state + 0.5 * dt * k1, control)
        k3 = self.derivatives(state + 0.5 * dt * k2, control)
        k4 = self.derivatives(state + dt * k3, control)
        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if self.state[1] < 0.0:
            self.state[1] = 0.0
            self.state[5] = 0.0
        return self.state.copy()

    def derivatives(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _, _, theta, alpha, vx, vz, omega, alpha_dot = state
        thrust, torque = control
        cfg = self.config

        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        matrix = np.array(
            [
                [cfg.total_mass, 0.0, self._mass_rope * cos_alpha],
                [0.0, cfg.total_mass, self._mass_rope * sin_alpha],
                [
                    self._mass_rope * cos_alpha,
                    self._mass_rope * sin_alpha,
                    cfg.payload_inertia,
                ],
            ],
            dtype=np.float64,
        )
        centrifugal = self._mass_rope * alpha_dot**2
        vector = np.array(
            [
                thrust * sin_theta + centrifugal * sin_alpha - cfg.drag_x * vx,
                thrust * cos_theta
                - cfg.total_mass * cfg.gravity
                - centrifugal * cos_alpha
                - cfg.drag_z * vz,
                -self._gravity_moment * sin_alpha - cfg.swing_damping * alpha_dot,
            ],
            dtype=np.float64,
        )
        x_ddot, z_ddot, alpha_ddot = _solve_3x3(matrix, vector)
        theta_ddot = (torque - cfg.drag_pitch * omega) / cfg.pitch_inertia
        return np.array(
            [vx, vz, omega, alpha_dot, x_ddot, z_ddot, theta_ddot, alpha_ddot],
            dtype=np.float64,
        )
