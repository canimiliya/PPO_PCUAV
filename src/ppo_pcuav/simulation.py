from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from .environment import PayloadUAVEnv, Scenario
from .planning import (
    AStarPlanner,
    RRTStarPlanner,
    build_wire_obstacles,
    smooth_path,
)


@dataclass
class FlightLog:
    label: str
    states: np.ndarray
    times: np.ndarray
    success: bool

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.states[:, :2], axis=0), axis=1).sum())

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def max_swing_degrees(self) -> float:
        return float(np.max(np.abs(np.degrees(self.states[:, 3]))))

    @property
    def swing_intensity(self) -> float:
        return float(np.trapezoid(self.states[:, 3] ** 2, self.times))


def run_policy(model: PPO, scenario: Scenario) -> FlightLog:
    environment = PayloadUAVEnv()
    observation, _ = environment.reset(seed=0, options={"scenario": scenario})
    states = [environment.state.copy()]
    success = False
    arrival_hold = 0
    required_hold = int(0.5 / environment.dynamics.dt)

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        step_states = info.get("step_trajectory")
        new_states = step_states[1:] if step_states is not None else [environment.state.copy()]
        for state in new_states:
            states.append(np.asarray(state, dtype=np.float64))
            distance = np.linalg.norm(environment.target_position - state[:2])
            speed = np.linalg.norm(state[4:6])
            arrival_hold = arrival_hold + 1 if distance < 0.10 and speed < 0.05 else 0
            if arrival_hold >= required_hold:
                success = True
                break
        success = success or bool(info["is_success"])
        if success or terminated or truncated:
            break

    states_array = np.asarray(states, dtype=np.float64)
    times = np.arange(len(states_array), dtype=np.float64) * environment.dynamics.dt
    environment.close()
    return FlightLog("PPO-PID", states_array, times, success)


def run_baseline(
    planner_name: str,
    scenario: Scenario,
    seed: int = 0,
    maximum_time: float = 60.0,
) -> FlightLog:
    environment = PayloadUAVEnv()
    environment.reset(seed=seed, options={"scenario": scenario})
    obstacles = build_wire_obstacles(environment, scenario.wire_index)
    bounds = (-10.0, 60.0), (2.5, 60.0)
    if planner_name == "A*":
        planner = AStarPlanner(*bounds, obstacles)
        raw_path = planner.plan(environment.start_position, environment.target_position)
    elif planner_name == "RRT*":
        planner = RRTStarPlanner(*bounds, obstacles)
        raw_path = planner.plan(
            environment.start_position, environment.target_position, seed=seed
        )
    else:
        raise ValueError(f"Unsupported planner: {planner_name}")
    path = smooth_path(planner, raw_path)

    states = [environment.state.copy()]
    path_index = 0
    arrival_hold = 0
    required_hold = int(0.5 / environment.dynamics.dt)
    maximum_steps = int(maximum_time / environment.dynamics.dt)
    success = False
    settling = False

    for _ in range(maximum_steps):
        state = environment.state
        while path_index < len(path) - 1 and np.linalg.norm(state[:2] - path[path_index]) < 1.0:
            path_index += 1
        target = path[path_index] if not settling else environment.target_position
        if settling:
            target_velocity = tuple(np.clip(-0.6 * state[4:6], -0.8, 0.8))
        else:
            target_velocity = (0.0, 0.0)
        control = environment.controller.compute_control(state, target, target_velocity)
        environment.dynamics.step(control)
        states.append(environment.state.copy())

        crashed, _ = environment.check_collision(environment.state[:2])
        if crashed:
            success = False
            break
        distance = np.linalg.norm(environment.target_position - environment.state[:2])
        speed = np.linalg.norm(environment.state[4:6])
        if distance < 0.8 and speed < 0.5:
            settling = True
        arrival_hold = arrival_hold + 1 if distance < 0.10 and speed < 0.05 else 0
        if arrival_hold >= required_hold:
            success = True
            break

    states_array = np.asarray(states, dtype=np.float64)
    times = np.arange(len(states_array), dtype=np.float64) * environment.dynamics.dt
    environment.close()
    return FlightLog(f"{planner_name}-PID", states_array, times, success)


def metrics(logs: list[FlightLog]) -> list[dict]:
    return [
        {
            "method": log.label,
            "success": log.success,
            "duration_s": round(log.duration, 3),
            "path_length_m": round(log.path_length, 3),
            "max_swing_deg": round(log.max_swing_degrees, 3),
            "swing_intensity_rad2_s": round(log.swing_intensity, 4),
        }
        for log in logs
    ]
