from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DroneConfig:
    """Physical parameters of the FC30 UAV and its suspended payload."""

    gravity: float = 9.81
    drone_mass: float = 65.0
    payload_mass: float = 30.0
    arm_length: float = 1.1
    pitch_inertia: float = 9.2
    rope_length: float = 1.2
    payload_height: float = 0.9
    payload_width: float = 0.4
    drag_x: float = 0.2
    drag_z: float = 0.2
    drag_pitch: float = 1.0
    swing_damping: float = 0.5
    max_torque: float = 120.0

    @property
    def max_thrust(self) -> float:
        return self.drone_mass * self.gravity * 2.0

    @property
    def max_angle(self) -> float:
        return float(np.radians(30.0))

    @property
    def effective_rope_length(self) -> float:
        return self.rope_length + self.payload_height / 2.0

    @property
    def payload_inertia(self) -> float:
        center = (self.payload_mass / 12.0) * (
            self.payload_height**2 + self.payload_width**2
        )
        return self.payload_mass * self.effective_rope_length**2 + center

    @property
    def total_mass(self) -> float:
        return self.drone_mass + self.payload_mass
