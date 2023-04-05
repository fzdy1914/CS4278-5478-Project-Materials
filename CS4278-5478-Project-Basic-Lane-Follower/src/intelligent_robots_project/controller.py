"""This module provides low-level control and planning functionality."""
import typing as T
import logging

import numpy as np
from scipy import interpolate


logger = logging.getLogger(__name__)


def sign(val):
    return -1 if val < 0 else 1


def convert_tile_traj_to_bot_frame(local_state, traj):
    R_robot_tile = np.array(
        [
            [np.cos(local_state[2]), np.sin(local_state[2])],
            [-np.sin(local_state[2]), np.cos(local_state[2])],
        ]
    )
    T_robot_tile = np.eye(3)
    T_robot_tile[:2, :2] = R_robot_tile
    T_robot_tile[:2, 2] = -R_robot_tile @ local_state[:2]
    traj = np.vstack([traj, np.ones(traj.shape[1])])
    traj = T_robot_tile @ traj
    return traj[0:2, :]


class SimpleController:
    def __init__(
        self,
        Kv=3,
        Kw=5,
        Dw=0.01,
        v_max=1,
        w_max=10,
        slowdown_factor=0.1,
        look_ahead_distance=0.1,
    ):
        self.Kv = Kv
        self.Kw = Kw
        self.Dw = Dw
        self.v_max = v_max
        self.w_max = w_max
        self.look_ahead_distance = look_ahead_distance
        self._prev_err_theta = None
        self._slowdown_factor = slowdown_factor
        self._tile_waypoints = None
        self._wp_idx = 0

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def set_trajectory(self, tile_waypoints):
        self._tile_waypoints = tile_waypoints
        self._wp_idx = 0

    def compute(self, state):
        if self._tile_waypoints is None:
            raise RuntimeError("No trajectory specified!")
        reference = convert_tile_traj_to_bot_frame(state, self._tile_waypoints)
        while True:
            if self._wp_idx >= reference.shape[1] - 1:
                self._wp_idx = reference.shape[1] - 1
                break
            if np.linalg.norm(reference[:, self._wp_idx]) > self.look_ahead_distance:
                self._wp_idx += 1
                break
            self._wp_idx += 1
        target_x = reference[0][self._wp_idx]
        target_y = reference[1][self._wp_idx]
        logger.debug(
            "Controller reference (target_x, target_y) = (%s, %s)", target_x, target_y
        )
        err_dist = np.linalg.norm(np.array([target_x, target_y]))
        if err_dist <= self.look_ahead_distance / 5.0:
            logger.debug("To close to the look ahead point. Stay still.")
            return [0, 0]
        err_theta = np.arctan2(target_y, target_x)
        derr_theta = (
            0 if self._prev_err_theta is None else (self._prev_err_theta - err_theta)
        )
        omega = -(
            self.Kw * err_theta + self.Dw * derr_theta
        )  # positive command corresponds to negative z rotation
        if abs(omega) > self.w_max:
            omega = self.w_max * sign(omega)
        v = min(self.Kv * err_dist, self.v_max)
        v = max(v - self._slowdown_factor * abs(err_theta), 0.02)
        logger.debug("Controller action (v, w) = (%s, %s)", v, omega)
        self._prev_err_theta = err_theta
        return [v, omega]


class TrajectoryGenerator:
    """Generate trajectories from local state and command."""

    def __init__(self, y_ref=0.70):
        self._y_ref = y_ref

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, state, command) -> T.List[T.List[float]]:
        """Return a list of waypoints in bot's frame.

        Returns:
            A list of lists. The first row corresponds to x and the second to y.
        """
        logger.debug(
            "Generating trajectory from local state %s and command %s", state, command
        )
        if command == "forward" or command is None:
            cur = np.array(state[0:2])
            waypoints = [cur]
            x1 = min(state[0] + 0.1, 2.0)
            x2 = x1 + 1
            waypoints.append(np.array([x1, self._y_ref]))
            waypoints.append(np.array([x2, self._y_ref]))
            logger.debug("Generated waypoints %s", waypoints)
            segments = []
            for i in range(len(waypoints) - 1):
                segments.append(
                    np.linspace(waypoints[i], waypoints[i + 1], num=15, endpoint=False)
                )
            if len(segments) > 1:
                traj = np.concatenate(segments).T
            else:
                traj = segments[0].T
        elif command == "left":
            cur = np.array(state[0:2])
            waypoints = [cur]
            if cur[0] < 0.25:
                waypoints.append(np.array([0.3, self._y_ref]))
            if cur[1] > 0.35:
                waypoints.append(np.array([self._y_ref, 0.3]))
            waypoints.append(np.array([self._y_ref, -0.1]))
            waypoints.append(np.array([self._y_ref, -0.5]))
            logger.debug("Generated waypoints %s", waypoints)
            segments = []
            for i in range(len(waypoints) - 1):
                segments.append(
                    np.linspace(waypoints[i], waypoints[i + 1], num=15, endpoint=False)
                )
            if len(segments) > 1:
                traj = np.concatenate(segments).T
            else:
                traj = segments[0].T
        elif command == "right":
            cur = np.array(state[0:2])
            waypoints = [cur]
            if cur[1] < 0.85:
                waypoints.append(np.array([0.25, 0.8]))
            if cur[1] < 1:
                waypoints.append(np.array([1 - self._y_ref, 1.1]))
            waypoints.append(np.array([1 - self._y_ref, 1.5]))
            logger.debug("Generated waypoints %s", waypoints)
            segments = []
            for i in range(len(waypoints) - 1):
                segments.append(
                    np.linspace(waypoints[i], waypoints[i + 1], num=15, endpoint=False)
                )
            if len(segments) > 1:
                traj = np.concatenate(segments).T
            else:
                traj = segments[0].T
        elif command == "uturn":
            cur = np.array(state[0:2])
            waypoints = [cur]
            if cur[1] > 0.5:
                if cur[0] < 0.2:
                    waypoints.append(np.array([0.3, self._y_ref]))
                waypoints.append(np.array([0.5, 0.5]))
            if cur[0] > 0:
                if cur[0] > 0.2:
                    waypoints.append(np.array([0.1, 1 - self._y_ref]))
                waypoints.append(np.array([-0.1, 1 - self._y_ref]))
            waypoints.append(np.array([-1.5, 1 - self._y_ref]))
            logger.debug("Generated waypoints %s", waypoints)
            segments = []
            for i in range(len(waypoints) - 1):
                segments.append(
                    np.linspace(waypoints[i], waypoints[i + 1], num=15, endpoint=False)
                )
            if len(segments) > 1:
                traj = np.concatenate(segments).T
            else:
                traj = segments[0].T
        else:
            logger.warning("Command %s not implemented!", command)
            raise NotImplementedError("Command %s not implemented!" % command)
        tck, _ = interpolate.splprep(traj)
        s = np.arange(0, 1.01, 0.01)
        traj = interpolate.splev(s, tck)
        traj = np.array(traj)
        return traj
