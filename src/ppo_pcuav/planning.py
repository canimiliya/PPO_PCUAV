import heapq
import math

import numpy as np
from scipy.interpolate import splev, splprep

from .environment import PayloadUAVEnv


class PathPlanner:
    def __init__(
        self,
        x_range: tuple[float, float],
        z_range: tuple[float, float],
        obstacles: list[tuple[float, float, float]],
        safety_margin: float = 0.5,
    ):
        self.x_min, self.x_max = x_range
        self.z_min, self.z_max = z_range
        self.obstacles = list(obstacles)
        self.safety_margin = safety_margin

    def collides(self, point: np.ndarray | tuple[float, float]) -> bool:
        x, z = float(point[0]), float(point[1])
        if not (self.x_min <= x <= self.x_max and self.z_min <= z <= self.z_max):
            return True
        return any(
            math.hypot(x - ox, z - oz) <= radius + self.safety_margin
            for ox, oz, radius in self.obstacles
        )

    def segment_collides(self, start, end, spacing: float = 0.25) -> bool:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / spacing)) + 1)
        return any(self.collides(start + t * (end - start)) for t in np.linspace(0.0, 1.0, count))


class AStarPlanner(PathPlanner):
    """Eight-connected A* planner on a 0.5 m grid."""

    resolution = 0.5
    motions = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    def plan(self, start, goal) -> np.ndarray:
        def to_grid(point):
            return (
                int(round((point[0] - self.x_min) / self.resolution)),
                int(round((point[1] - self.z_min) / self.resolution)),
            )

        def to_position(node):
            return np.array(
                [
                    node[0] * self.resolution + self.x_min,
                    node[1] * self.resolution + self.z_min,
                ],
                dtype=np.float64,
            )

        start_node, goal_node = to_grid(start), to_grid(goal)
        queue = [(0.0, start_node)]
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start_node: 0.0}

        while queue:
            _, current = heapq.heappop(queue)
            if current == goal_node:
                nodes = [current]
                while current in parent:
                    current = parent[current]
                    nodes.append(current)
                path = np.asarray([to_position(node) for node in reversed(nodes)])
                path[0] = start
                path[-1] = goal
                return path

            for dx, dz in self.motions:
                neighbor = current[0] + dx, current[1] + dz
                position = to_position(neighbor)
                if self.collides(position):
                    continue
                tentative = cost[current] + math.hypot(dx, dz)
                if tentative >= cost.get(neighbor, float("inf")):
                    continue
                parent[neighbor] = current
                cost[neighbor] = tentative
                heuristic = math.hypot(
                    neighbor[0] - goal_node[0], neighbor[1] - goal_node[1]
                )
                heapq.heappush(queue, (tentative + heuristic, neighbor))
        raise RuntimeError("A* could not find a collision-free path")


class RRTStarPlanner(PathPlanner):
    """Seeded RRT* planner with collision-checked parent selection and rewiring."""

    def plan(
        self,
        start,
        goal,
        seed: int = 0,
        max_iterations: int = 3000,
        step_size: float = 1.0,
        search_radius: float = 5.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        tree = [start]
        parents: dict[int, int | None] = {0: None}
        costs = {0: 0.0}

        for _ in range(max_iterations):
            sample = (
                goal
                if rng.random() < 0.1
                else np.array(
                    [rng.uniform(self.x_min, self.x_max), rng.uniform(self.z_min, self.z_max)]
                )
            )
            nearest_index = int(
                np.argmin([np.linalg.norm(point - sample) for point in tree])
            )
            nearest = tree[nearest_index]
            direction = sample - nearest
            distance = float(np.linalg.norm(direction))
            if distance < 1e-12:
                continue
            new_point = nearest + step_size * direction / distance
            if self.collides(new_point) or self.segment_collides(nearest, new_point):
                continue

            neighbors = [
                index
                for index, point in enumerate(tree)
                if np.linalg.norm(point - new_point) <= search_radius
            ]
            parent_index = nearest_index
            best_cost = costs[nearest_index] + float(np.linalg.norm(new_point - nearest))
            for index in neighbors:
                candidate_cost = costs[index] + float(np.linalg.norm(new_point - tree[index]))
                if candidate_cost < best_cost and not self.segment_collides(tree[index], new_point):
                    parent_index, best_cost = index, candidate_cost

            tree.append(new_point)
            new_index = len(tree) - 1
            parents[new_index], costs[new_index] = parent_index, best_cost
            for index in neighbors:
                candidate_cost = best_cost + float(np.linalg.norm(tree[index] - new_point))
                if candidate_cost < costs[index] and not self.segment_collides(new_point, tree[index]):
                    parents[index], costs[index] = new_index, candidate_cost

            if np.linalg.norm(new_point - goal) < step_size and not self.segment_collides(new_point, goal):
                path = [goal, new_point]
                current = new_index
                while parents[current] is not None:
                    current = int(parents[current])
                    path.append(tree[current])
                return np.asarray(list(reversed(path)), dtype=np.float64)
        raise RuntimeError("RRT* could not find a collision-free path")


def build_wire_obstacles(
    environment: PayloadUAVEnv, active_wire_index: int
) -> list[tuple[float, float, float]]:
    obstacles = []
    for index, (wire_x, wire_z) in enumerate(environment.wires):
        extra_buffer = 0.0 if index == active_wire_index else 0.8
        radius = max(0.1, environment.penalty_threshold + extra_buffer - 0.5)
        obstacles.append(
            (
                float(wire_x - environment.collision_offset[0]),
                float(wire_z - environment.collision_offset[1]),
                radius,
            )
        )
    return obstacles


def _polyline_is_safe(
    planner: PathPlanner, points: np.ndarray, spacing: float = 0.25
) -> bool:
    return all(
        not planner.segment_collides(points[index], points[index + 1], spacing)
        for index in range(len(points) - 1)
    )


def smooth_path(planner: PathPlanner, points: np.ndarray, count: int = 160) -> np.ndarray:
    """Use the smoothest collision-free quadratic B-spline, then densify as fallback."""

    points = np.asarray(points, dtype=np.float64)
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
    points = points[keep]
    if len(points) >= 3:
        for smoothing in (0.5, 0.2, 0.08, 0.02, 0.0):
            try:
                spline, _ = splprep(points.T, s=smoothing, k=2)
                candidate = np.column_stack(splev(np.linspace(0.0, 1.0, count), spline))
                if _polyline_is_safe(planner, candidate):
                    return candidate
            except ValueError:
                continue

    dense = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        samples = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.25)) + 1)
        dense.extend(start + t * (end - start) for t in np.linspace(0.0, 1.0, samples)[1:])
    return np.asarray(dense, dtype=np.float64)
