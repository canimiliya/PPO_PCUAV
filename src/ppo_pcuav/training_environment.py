import gymnasium as gym
import numpy as np

from .environment import PayloadUAVEnv


class CurriculumPayloadUAVEnv(PayloadUAVEnv):
    """The 26-level curriculum environment used to train the PPO policy.

    The final-level observation and action contracts are identical to
    :class:`PayloadUAVEnv`; only scenario sampling, success thresholds, and the
    shaped training reward are added here. Keeping training and deterministic
    demo environments separate prevents evaluation behavior from drifting.
    """

    def __init__(self):
        super().__init__()
        self.level = 1
        self.record_trajectory = False
        self.task_mode = "default"
        self.target_wire_index = 0
        self.fixed_task_mode: str | None = None
        self.wire_to_ground_probability = 0.5
        self.print_perfect_landing = False
        self._episode_start: np.ndarray | None = None
        self._line_direction: np.ndarray | None = None
        self._previous_line_progress: float | None = None

    def set_level(self, level: int) -> None:
        self.level = int(np.clip(level, 1, 26))

    def set_fixed_task_mode(self, mode: str | None) -> None:
        if mode not in {None, "ground_to_wire", "wire_to_ground"}:
            raise ValueError(f"Invalid task mode: {mode}")
        self.fixed_task_mode = mode

    def set_task_mode_probability(self, wire_to_ground: float) -> None:
        self.wire_to_ground_probability = float(np.clip(wire_to_ground, 0.0, 1.0))

    def set_print_perfect_landing(self, enabled: bool) -> None:
        self.print_perfect_landing = bool(enabled)

    def success_thresholds(
        self,
    ) -> tuple[float, float | None, float | None, float | None, float | None, float | None]:
        """Return distance, speed, swing, swing-rate, and bonus thresholds."""

        if self.level <= 15:
            return 0.25, None, None, None, 10.0, 0.20
        if self.level == 16:
            return 0.20, 0.08, None, None, 7.5, 0.15
        if self.level == 17:
            return 0.15, 0.06, None, None, 5.0, 0.10
        if self.level == 18:
            return 0.10, 0.04, None, None, 2.5, 0.05
        if self.level == 19:
            return 0.10, 0.04, 2.5, None, None, 0.05
        if self.level == 20:
            return 0.50, 0.10, None, None, 10.0, 0.20
        if self.level in {21, 22}:
            return 0.20, 0.08, None, None, 7.5, 0.15
        if self.level == 23:
            return 0.15, 0.06, None, None, 5.0, 0.10
        if self.level == 24:
            return 0.10, 0.04, None, None, 2.5, 0.05
        if self.level == 25:
            return 0.10, 0.04, 2.5, None, None, 0.05
        return 0.10, 0.04, 2.5, 0.05, None, None

    def reset(self, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        self._sample_wires()

        forced_wire = options.get("target_wire_index", options.get("target_wire_idx"))
        if forced_wire is None:
            wire_index = self._sample_target_wire()
        else:
            wire_index = int(np.clip(int(forced_wire), 0, len(self.wires) - 1))
        self.target_wire_index = wire_index

        wire_goal = self.wires[wire_index] + np.array([0.0, self.wire_goal_offset])
        if self.level >= 20:
            self._reset_bidirectional(options, wire_goal)
        else:
            self._reset_approach_curriculum(wire_goal)

        return self._observation(), {
            "level": self.level,
            "task_mode": self.task_mode,
            "target_wire_index": self.target_wire_index,
        }

    def _sample_wires(self) -> None:
        if self.level <= 3:
            self.wires = np.array([(20.0, 26.0), (30.0, 26.0)], dtype=np.float64)
        elif self.level <= 5:
            self.wires = np.array(
                [(20.0, 26.0), (30.0, 26.0), (22.0, 40.0), (28.0, 40.0)],
                dtype=np.float64,
            )
        elif self.level <= 8:
            lower = self.np_random.uniform(25.0, 27.0)
            upper = self.np_random.uniform(39.0, 41.0)
            self.wires = np.array(
                [(20.0, lower), (30.0, lower), (22.0, upper), (28.0, upper)],
                dtype=np.float64,
            )
        else:
            lower = self.np_random.uniform(20.0, 32.0)
            upper = self.np_random.uniform(36.0, 45.0)
            self.wires = np.array(
                [(20.0, lower), (30.0, lower), (22.0, upper), (28.0, upper)],
                dtype=np.float64,
            )

    def _sample_target_wire(self) -> int:
        if self.level <= 4:
            return 0
        return int(self.np_random.integers(0, len(self.wires)))

    def _reset_bidirectional(self, options: dict, wire_goal: np.ndarray) -> None:
        forced_mode = options.get("task_mode")
        if forced_mode in {"ground_to_wire", "wire_to_ground"}:
            mode = forced_mode
        elif self.fixed_task_mode is not None:
            mode = self.fixed_task_mode
        else:
            probability = float(
                np.clip(
                    options.get(
                        "wire_to_ground_probability", self.wire_to_ground_probability
                    ),
                    0.0,
                    1.0,
                )
            )
            mode = (
                "wire_to_ground"
                if self.np_random.random() < probability
                else "ground_to_wire"
            )

        ground_x_min = float(options.get("ground_x_min", options.get("x_min", 0.0)))
        ground_x_max = float(options.get("ground_x_max", options.get("x_max", 50.0)))
        ground_z = float(options.get("ground_z", options.get("z_ground", 3.0)))
        ground_x = float(self.np_random.uniform(ground_x_min, ground_x_max))
        ground = np.array([ground_x, ground_z], dtype=np.float64)

        if mode == "ground_to_wire":
            for _ in range(200):
                ground[0] = self.np_random.uniform(ground_x_min, ground_x_max)
                if not self.check_collision(ground)[0]:
                    break
            else:
                ground = np.array([0.0, ground_z], dtype=np.float64)
            start, target = ground, wire_goal
        else:
            start, target = wire_goal, ground

        self.task_mode = mode
        self._initialize_episode(start, target, track_direct_path=True)

    def _reset_approach_curriculum(self, wire_goal: np.ndarray) -> None:
        self.task_mode = "default"
        target = wire_goal.astype(np.float64)
        base_distance = 3.0 + (self.level - 1) * 2.3
        start = np.array([0.0, 50.0], dtype=np.float64)
        for _ in range(100):
            if self.level <= 4:
                distance = self.np_random.uniform(2.0, base_distance)
                direction = 1.0 if self.np_random.random() > 0.5 else -1.0
                candidate = np.array(
                    [target[0] + direction * distance, target[1]], dtype=np.float64
                )
            elif self.level <= 10:
                distance = self.np_random.uniform(5.0, base_distance)
                angle = self.np_random.uniform(0.0, 2.0 * np.pi)
                candidate = target + distance * np.array(
                    [np.cos(angle), 0.6 * np.sin(angle)]
                )
            elif self.level <= 12:
                candidate = np.array(
                    [self.np_random.uniform(-10.0, 50.0), self.np_random.uniform(40.0, 55.0)]
                )
            else:
                jitter = 5.0 if self.level == 13 else 2.0 if self.level == 14 else 0.1
                candidate = np.array(
                    [self.np_random.uniform(-jitter, jitter), 50.0 + self.np_random.uniform(-jitter, jitter)]
                )
            candidate[1] = max(2.5, candidate[1])
            if not self.check_collision(candidate)[0]:
                start = candidate
                break
        self._initialize_episode(start, target, track_direct_path=False)

    def _initialize_episode(
        self, start: np.ndarray, target: np.ndarray, *, track_direct_path: bool
    ) -> None:
        self.start_position = np.asarray(start, dtype=np.float64).copy()
        self.target_position = np.asarray(target, dtype=np.float64).copy()
        self.dynamics.reset([*self.start_position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.controller.reset()
        self.current_decision = 0
        self.previous_distance = float(
            np.linalg.norm(self.target_position - self.start_position)
        )
        if track_direct_path:
            self._episode_start = self.start_position.copy()
            direction = self.target_position - self.start_position
            norm = float(np.linalg.norm(direction))
            self._line_direction = direction / norm if norm > 1e-8 else None
            self._previous_line_progress = 0.0 if self._line_direction is not None else None
        else:
            self._episode_start = None
            self._line_direction = None
            self._previous_line_progress = None

    def step(self, action):
        indices = np.asarray(action, dtype=np.int64).reshape(2)
        distance = self.distance_options[indices[0]]
        angle = np.radians(self.angle_options[indices[1]])
        waypoint = self.state[:2] + distance * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )

        trajectory = [self.state.copy()] if self.record_trajectory else None
        crashed = False
        for _ in range(self.frame_skip):
            control = self.controller.compute_control(self.state, waypoint)
            self.dynamics.step(control)
            if trajectory is not None:
                trajectory.append(self.state.copy())
            crashed, _ = self.check_collision(self.state[:2])
            if crashed:
                break

        reward, terminated = self._training_reward(crashed)
        self.current_decision += 1
        truncated = bool(self.current_decision >= self.max_decisions and not terminated)
        success = self._is_success(crashed)
        info = {
            "is_success": success,
            "level": self.level,
            "task_mode": self.task_mode,
            "target_wire_index": self.target_wire_index,
        }
        if trajectory is not None:
            info["planned_waypoint"] = waypoint.astype(np.float32)
            info["step_trajectory"] = np.asarray(trajectory, dtype=np.float32)
        return self._observation(), float(reward), bool(terminated), truncated, info

    def _is_success(self, crashed: bool) -> bool:
        distance = float(np.linalg.norm(self.target_position - self.state[:2]))
        speed = float(np.linalg.norm(self.state[4:6]))
        swing_degrees = float(np.degrees(abs(self.state[3])))
        swing_rate = float(abs(self.state[7]))
        dist_limit, speed_limit, swing_limit, rate_limit, _, _ = self.success_thresholds()
        return bool(
            not crashed
            and distance < dist_limit
            and (speed_limit is None or speed < speed_limit)
            and (swing_limit is None or swing_degrees < swing_limit)
            and (rate_limit is None or swing_rate < rate_limit)
        )

    def _wire_geometry(self, position: np.ndarray) -> tuple[float, np.ndarray | None]:
        if len(self.wires) == 0:
            return float("inf"), None
        center = self.collision_center(position)
        distances = np.linalg.norm(self.wires - center, axis=1)
        index = int(np.argmin(distances))
        return float(distances[index]), self.wires[index]

    def _training_reward(self, crashed: bool) -> tuple[float, bool]:
        state = self.state
        position = state[:2]
        distance = float(np.linalg.norm(self.target_position - position))
        speed = float(np.linalg.norm(state[4:6]))
        swing = float(abs(state[3]))
        swing_rate = float(abs(state[7]))

        progress_reward = 2.0 * (self.previous_distance - distance) - 0.005 * distance
        self.previous_distance = distance

        obstacle_reward = 0.0
        if not crashed:
            _, clearance = self.check_collision(position)
            if clearance < 0.0:
                proximity = abs(clearance) / self.repulsion_radius
                obstacle_reward -= 60.0 * proximity**2

        safety_reward = 0.0
        minimum_wire_distance, closest_wire = self._wire_geometry(position)
        if minimum_wire_distance < self.penalty_threshold:
            proximity = float(
                np.clip(
                    (self.penalty_threshold - minimum_wire_distance)
                    / self.repulsion_radius,
                    0.0,
                    1.0,
                )
            )
            if closest_wire is not None:
                center = self.collision_center(position)
                away = center - closest_wire
                norm = float(np.linalg.norm(away))
                if norm > 1e-8:
                    approach_speed = max(0.0, -float(np.dot(state[4:6], away / norm)))
                    safety_reward -= 40.0 * approach_speed * proximity
            if state[5] < -0.10:
                safety_reward -= 60.0 * (-float(state[5])) * proximity

        efficiency_reward = 0.0
        if (
            self._episode_start is not None
            and self._line_direction is not None
            and self._previous_line_progress is not None
        ):
            offset = position - self._episode_start
            line_progress = float(np.dot(offset, self._line_direction))
            progress_delta = line_progress - self._previous_line_progress
            self._previous_line_progress = line_progress
            perpendicular = offset - line_progress * self._line_direction
            clearance = minimum_wire_distance - self.penalty_threshold
            gate = float(np.clip(clearance, 0.0, 1.0))
            efficiency_reward -= 0.03 * gate * float(np.linalg.norm(perpendicular)) ** 2
            efficiency_reward += 0.8 * gate * progress_delta

        swing_degrees = float(np.degrees(swing))
        swing_reward = -1.5 * (swing_degrees / 20.0) ** 2 - 0.8 * swing_rate**2
        if swing > np.radians(28.0):
            swing_reward -= 3.0

        distance_limit, _, _, _, bonus_swing, bonus_rate = self.success_thresholds()
        goal_band = float(np.clip(distance_limit * 6.0, 0.8, 2.0))
        goal_gate = float(np.clip((goal_band - distance) / goal_band, 0.0, 1.0))
        settling_reward = -goal_gate * (
            6.0 * speed**2 + 5.0 * swing**2 + 1.2 * swing_rate**2
        )
        total = (
            progress_reward
            + obstacle_reward
            + safety_reward
            + efficiency_reward
            + swing_reward
            - 0.45
            + settling_reward
        )

        out_of_bounds = position[1] > 60.0 or position[0] < -20.0 or position[0] > 60.0
        if crashed:
            return (-40.0 if self.level >= 20 else -20.0) * 0.1 - distance * 0.1, True
        if out_of_bounds:
            return (-20.0 - distance) * 0.1, True
        if not self._is_success(False):
            return total * 0.1, False

        success_reward = 95.0 if self.level == 26 else 80.0
        bonus_ok = (
            (bonus_swing is None or swing_degrees < bonus_swing)
            and (bonus_rate is None or swing_rate < bonus_rate)
        )
        if bonus_ok and (bonus_swing is not None or bonus_rate is not None):
            success_reward += 120.0
        success_reward += 0.5 * self.level
        if self.level == 26 and self.print_perfect_landing:
            print(
                f"L26 landing: distance={distance:.4f}, speed={speed:.4f}, "
                f"swing={swing_degrees:.3f} deg, swing_rate={swing_rate:.4f}"
            )
        return success_reward * 0.1, True
