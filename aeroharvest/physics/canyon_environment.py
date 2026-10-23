"""
Canyon Environment & Topography for AeroHarvest Simulation.
Models a 1 km x 1 km x 300 m Mountain Canyon with realistic valley slopes,
thermal inspection target areas, and perimeter waypoints.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class ThermalTarget:
    target_id: int
    position: np.ndarray  # [x, y, z] in meters
    radius: float
    required_dwell_time: float
    current_dwell_time: float = 0.0
    completed: bool = False

    def is_inside(self, pos: np.ndarray) -> bool:
        dist = np.linalg.norm(pos - self.position)
        return dist <= self.radius


class MountainCanyon:
    """
    Continuous elevation and obstacle model for a 1 km x 1 km mountain canyon.
    Origin (0,0) is centered in the valley.
    Domain bounds: x in [-500, 500], y in [-500, 500], z in [0, 300].
    """

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-500.0, 500.0),
        bounds_y: Tuple[float, float] = (-500.0, 500.0),
        bounds_z: Tuple[float, float] = (0.0, 300.0),
    ):
        self.x_min, self.x_max = bounds_x
        self.y_min, self.y_max = bounds_y
        self.z_min, self.z_max = bounds_z

        # Canyon trench orientation: primary channel curves through the canyon
        # We generate procedural canyon terrain using layered harmonic functions
        self._canyon_width = 160.0  # meters width of floor
        self._canyon_depth = 120.0  # meters peak-to-trough
        self._ridge_freq = 0.005

    def get_elevation(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Computes terrain height z_terrain(x, y) in meters.
        Features a deep canyon running roughly SW to NE with side ridges.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # Centerline of the canyon follows a gentle S-curve
        y_center = 80.0 * np.sin(0.004 * x_arr)
        dist_from_canyon = np.abs(y_arr - y_center)

        # Base valley floor height ~15m
        base_h = 15.0

        # Valley profile: smooth tanh canyon walls
        wall_height = 85.0 * (1.0 + np.tanh((dist_from_canyon - self._canyon_width * 0.5) / 50.0))

        # Mountain ridgelines and lateral peaks
        ridges = (
            25.0 * np.cos(self._ridge_freq * x_arr) * np.sin(self._ridge_freq * y_arr * 1.5)
            + 15.0 * np.sin(0.012 * x_arr + 0.008 * y_arr)
            + 10.0 * np.cos(0.02 * y_arr)
        )

        elevation = base_h + wall_height + ridges
        elevation = np.clip(elevation, 0.0, self.z_max - 20.0)

        if np.isscalar(x) and np.isscalar(y):
            return float(elevation)
        return elevation

    def get_altitude_above_ground(self, position: np.ndarray) -> float:
        """Returns height above terrain for a 3D coordinate [x, y, z]."""
        terrain_z = self.get_elevation(position[0], position[1])
        return position[2] - terrain_z

    def check_collision(self, position: np.ndarray, min_clearance: float = 5.0) -> bool:
        """True if position is within minimum clearance of terrain or outside bounds."""
        if (
            position[0] < self.x_min
            or position[0] > self.x_max
            or position[1] < self.y_min
            or position[1] > self.y_max
            or position[2] < self.z_min
            or position[2] > self.z_max
        ):
            return True
        return self.get_altitude_above_ground(position) < min_clearance

    @staticmethod
    def centerline_y(x: float) -> float:
        """Y coordinate of the canyon trench centerline at a given x."""
        return float(80.0 * np.sin(0.004 * x))

    def point_in_canyon(self, x: float, lateral_offset: float, agl: float) -> np.ndarray:
        """3D point at `lateral_offset` m from the centerline and `agl` m above terrain."""
        y = self.centerline_y(x) + lateral_offset
        z = self.get_elevation(x, y) + agl
        return np.array([x, y, z], dtype=np.float64)


class MissionScenario:
    """
    Defines the 6-waypoint perimeter inspection task and 3 thermal inspection targets
    specified in Section 6.1 and 10 of the PRD.
    """

    # Mission legs follow the canyon trench so every point is above terrain:
    # (x [m], lateral offset from canyon centerline [m], altitude above ground [m])
    LAUNCH_LEG = (-430.0, 0.0, 5.0)
    MISSION_LEGS = [
        (-300.0, 30.0, 45.0),
        (-150.0, -35.0, 40.0),  # Thermal Target 1
        (0.0, 35.0, 45.0),      # Thermal Target 2
        (150.0, -30.0, 50.0),
        (300.0, 35.0, 40.0),    # Thermal Target 3
        (430.0, 0.0, 50.0),
    ]
    TARGETS = [  # (target_id, mission leg index, radius [m], dwell [s])
        (1, 1, 40.0, 8.0),
        (2, 2, 45.0, 10.0),
        (3, 4, 40.0, 8.0),
    ]

    def __init__(self, canyon: MountainCanyon):
        self.canyon = canyon
        self.launch_point = canyon.point_in_canyon(*self.LAUNCH_LEG)

        # 6-waypoint inspection route along the canyon floor
        self.perimeter_waypoints = [canyon.point_in_canyon(*leg) for leg in self.MISSION_LEGS]

        # 3 thermal inspection zones co-located with mission waypoints
        self.thermal_targets: List[ThermalTarget] = [
            ThermalTarget(
                target_id=tid,
                position=self.perimeter_waypoints[leg_idx].copy(),
                radius=radius,
                required_dwell_time=dwell,
            )
            for tid, leg_idx, radius, dwell in self.TARGETS
        ]

        self.current_waypoint_idx = 0
        self.waypoint_acceptance_radius = 20.0

    def update_targets(self, position: np.ndarray, dt: float) -> List[int]:
        """
        Updates dwell time for targets within radius.
        Returns list of newly completed target IDs.
        """
        completed_ids = []
        for target in self.thermal_targets:
            if not target.completed and target.is_inside(position):
                target.current_dwell_time += dt
                if target.current_dwell_time >= target.required_dwell_time:
                    target.completed = True
                    completed_ids.append(target.target_id)
        return completed_ids

    def get_active_thermal_target(self, position: np.ndarray) -> ThermalTarget | None:
        """Returns the thermal target if drone is within inspection radius."""
        for target in self.thermal_targets:
            if not target.completed and target.is_inside(position):
                return target
        return None

    def get_current_waypoint(self) -> np.ndarray:
        if self.current_waypoint_idx < len(self.perimeter_waypoints):
            return self.perimeter_waypoints[self.current_waypoint_idx]
        return self.launch_point

    def check_waypoint_reached(self, position: np.ndarray) -> bool:
        target_wp = self.get_current_waypoint()
        dist = np.linalg.norm(position[:2] - target_wp[:2])
        z_dist = abs(position[2] - target_wp[2])
        if dist <= self.waypoint_acceptance_radius and z_dist <= 15.0:
            if self.current_waypoint_idx < len(self.perimeter_waypoints):
                self.current_waypoint_idx += 1
                return True
        return False

    def is_mission_complete(self) -> bool:
        all_targets_done = all(t.completed for t in self.thermal_targets)
        all_waypoints_done = self.current_waypoint_idx >= len(self.perimeter_waypoints)
        return all_targets_done and all_waypoints_done
