from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DroneConfig
from .controller import CascadedPIDController
from .dynamics import PayloadUAVDynamics


CANONICAL_WIRES = np.array(
    [(20.0, 26.0), (30.0, 26.0), (22.0, 40.0), (28.0, 40.0)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Scenario:
    mode: str = "ground_to_wire"
    wire_index: int = 0
    ground_x: float = 45.0
    ground_z: float = 3.0

    def __post_init__(self) -> None:
        if self.mode not in {"ground_to_wire", "wire_to_ground"}:
            raise ValueError(f"Unsupported scenario mode: {self.mode}")
        if self.wire_index not in range(4):
            raise ValueError("wire_index must be between 0 and 3")


class PayloadUAVEnv(gym.Env):
    """Inference environment matching the observation/action contract of the trained policy."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.config = DroneConfig()
        self.dynamics = PayloadUAVDynamics(self.config)
        self.controller = CascadedPIDController(self.config)

        self.action_space = spaces.MultiDiscrete([9, 24])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.distance_options = np.array(
            [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2],
            dtype=np.float32,
        )
        self.angle_options = np.arange(0.0, 360.0, 15.0, dtype=np.float32)

        self.frame_skip = 20
        self.max_decisions = 125
        self.wire_goal_offset = 3.48
        self.collision_offset = np.array([0.0, -0.5], dtype=np.float64)
        self.system_radius = 2.0
        self.wire_radius = 0.03
        self.repulsion_radius = 0.75
        self.crash_threshold = self.system_radius + self.wire_radius
        self.penalty_threshold = self.crash_threshold + self.repulsion_radius

        self.wires = CANONICAL_WIRES.copy()
        self.target_position = np.zeros(2, dtype=np.float64)
        self.start_position = np.zeros(2, dtype=np.float64)
        self.scenario = Scenario()
        self.current_decision = 0
        self.previous_distance = 0.0
        self.record_trajectory = True

    @property
    def state(self) -> np.ndarray:
        return self.dynamics.state

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario", Scenario())
        if isinstance(scenario, dict):
            scenario = Scenario(**scenario)
        if not isinstance(scenario, Scenario):
            raise TypeError("options['scenario'] must be a Scenario or a mapping")
        self.scenario = scenario
        self.wires = np.asarray(options.get("wires", CANONICAL_WIRES), dtype=np.float64).copy()

        wire = self.wires[scenario.wire_index]
        wire_goal = wire + np.array([0.0, self.wire_goal_offset])
        ground = np.array([scenario.ground_x, scenario.ground_z], dtype=np.float64)
        if scenario.mode == "ground_to_wire":
            self.start_position = ground
            self.target_position = wire_goal
        else:
            self.start_position = wire_goal
            self.target_position = ground

        self.dynamics.reset(
            [*self.start_position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.controller.reset()
        self.current_decision = 0
        self.previous_distance = float(
            np.linalg.norm(self.target_position - self.start_position)
        )
        return self._observation(), self._info(False, False)

    def step(self, action):
        indices = np.asarray(action, dtype=np.int64).reshape(2)
        move_distance = self.distance_options[indices[0]]
        move_angle = np.radians(self.angle_options[indices[1]])
        waypoint = self.state[:2] + move_distance * np.array(
            [np.cos(move_angle), np.sin(move_angle)], dtype=np.float64
        )

        trajectory = [self.state.copy()]
        crashed = False
        for _ in range(self.frame_skip):
            control = self.controller.compute_control(self.state, waypoint)
            self.dynamics.step(control)
            trajectory.append(self.state.copy())
            crashed, _ = self.check_collision(self.state[:2])
            if crashed:
                break

        self.current_decision += 1
        distance = float(np.linalg.norm(self.target_position - self.state[:2]))
        speed = float(np.linalg.norm(self.state[4:6]))
        success = (
            not crashed
            and distance < 0.10
            and speed < 0.04
            and abs(np.degrees(self.state[3])) < 2.5
            and abs(self.state[7]) < 0.05
        )
        out_of_bounds = (
            self.state[1] > 60.0 or self.state[0] < -20.0 or self.state[0] > 60.0
        )
        terminated = bool(crashed or out_of_bounds or success)
        truncated = bool(self.current_decision >= self.max_decisions and not terminated)

        reward = 0.2 * (self.previous_distance - distance) - 0.045
        self.previous_distance = distance
        if crashed or out_of_bounds:
            reward = -10.0 - 0.1 * distance
        elif success:
            reward = 10.0

        info = self._info(success, crashed)
        info["planned_waypoint"] = waypoint.astype(np.float32)
        if self.record_trajectory:
            info["step_trajectory"] = np.asarray(trajectory, dtype=np.float32)
        return self._observation(), float(reward), terminated, truncated, info

    def collision_center(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(position, dtype=np.float64) + self.collision_offset

    def check_collision(self, position: np.ndarray) -> tuple[bool, float]:
        center = self.collision_center(position)
        if center[1] < self.system_radius:
            return True, 0.0
        clearance = float("inf")
        for wire in self.wires:
            center_distance = float(np.linalg.norm(center - wire))
            if center_distance < self.crash_threshold:
                return True, 0.0
            clearance = min(clearance, center_distance - self.penalty_threshold)
        return False, clearance

    def _observation(self) -> np.ndarray:
        position = self.state[:2]
        relative_target = self.target_position - position
        relative_wires = [wire - position for wire in self.wires[:4]]
        while len(relative_wires) < 4:
            relative_wires.append(np.zeros(2, dtype=np.float64))
        raw = np.concatenate(
            [self.state, relative_target, np.concatenate(relative_wires)]
        )
        scale = np.array(
            [
                0.02,
                0.02,
                1.0,
                1.0,
                0.1,
                0.1,
                0.1,
                0.1,
                0.02,
                0.02,
                *([0.02, 0.02] * 4),
            ],
            dtype=np.float64,
        )
        return (raw * scale).astype(np.float32)

    def _info(self, success: bool, crashed: bool) -> dict:
        return {
            "is_success": bool(success),
            "crashed": bool(crashed),
            "task_mode": self.scenario.mode,
            "target_wire_index": self.scenario.wire_index,
        }
