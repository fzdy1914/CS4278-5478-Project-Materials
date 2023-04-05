"""This module implements the state estimator."""
import logging
import typing as T
from scipy.integrate import odeint
from scipy import stats

import numpy as np

logger = logging.getLogger(__name__)


def clip_angle(ang):
    if ang > np.pi:
        return ang - np.pi
    if ang < -np.pi:
        return ang + np.pi
    return ang


def calculate_global_orientation(prev, succ) -> T.Literal[0, 1, 2, 3]:
    """Calculated the global orientation.

    0: East
    1: North
    2: West
    3: South
    """
    logger.info("Estimation global orientation from %s to %s.", prev, succ)
    if prev[0] == succ[0]:
        if succ[1] == prev[1] + 1:
            return 1
        elif succ[1] == prev[1] - 1:
            return 3
        else:
            None
    elif prev[1] == succ[1]:
        if succ[0] == prev[0] + 1:
            return 0
        elif succ[0] == prev[0] - 1:
            return 2
        else:
            return None
    else:
        logger.warn("Previous tile %s and next tile %s are not adjacent.", prev, succ)


class Dynamics:
    """Nominal dynamic model of the system."""

    def __init__(self, dt=1.0 / 30.0):
        self._dt = dt

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, state, control):
        v, w = control
        y = odeint(self.f, np.array(state), np.array([0, self._dt]), args=(v, w))
        new_state = y[-1].tolist()
        new_state[2] = clip_angle(new_state[2])
        return new_state

    @staticmethod
    def f(state, t, v, w):
        x, y, theta = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w
        return np.array([dx, dy, dtheta])


class StateEstimator:
    """A class to estimate local and global states.


    Args:
        red_centroid_to_white: Distance of centroid of the stop line to its
            adjacent white line.
        red_centroid_to_white: Distance of centroid of the stop line to its
            adjacent junction boundary.
    """

    def __init__(
        self,
        init_local=None,
        init_global=None,
        dt=1 / 22.0,
        red_centroid_to_white=0.2,
        red_centroid_in_junction=0.05,
    ):
        self._local = init_local
        self._global = init_global
        self._f = Dynamics(dt=dt)
        self._current_node_kind = None
        self._current_command = None
        self._rotate_90 = np.array(
            [[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]
        )
        self._red_centroid_positions = [red_centroid_to_white, red_centroid_in_junction]
        self._stop_line_init_count = 0
        self.can_stop = False

    @property
    def global_state(self):
        return list(self._global)

    @property
    def local_state(self):
        if self._local is None:
            return None
        return list(self._local)

    def update_gps(self, gps: T.List[float]) -> bool:
        logger.debug("New GPS reading: %s", gps)
        if self._global is None:
            self._global = [gps[0], gps[1], None]
            logger.info("Update global state to %s", self._global)
            return True
        elif gps[0] != self._global[0] or gps[1] != self._global[1]:
            self._global[2] = calculate_global_orientation(self._global, gps)
            self._global[0] = gps[0]
            self._global[1] = gps[1]
            logger.info("Update global state to %s", self._global)
            self._update_local_state_from_gps(gps)
            logger.info("Update local state to %s", self._local)
            return True
        else:
            logger.debug("No global state change.")
            return False

    def _update_local_state_from_gps(self, gps: T.List[float]) -> bool:
        if self._local is None:
            self._local = [0, 0, 0]
        if self._current_command is None:
            self._local[0] = 0.0
        elif self._current_command == "forward":
            self._local[0] = 0.0
            self._local[2] = 0.0
        elif self._current_command == "left":
            self._local = [0.0, self._local[0], np.pi / 2 + self._local[2]]
        elif self._current_command == "right":
            self._local = [0.0, 1 - self._local[0], self._local[2] - np.pi / 2]
        elif self._current_command == "uturn":
            self._local = [0.0, 1 - self._local[1], self._local[2] + np.pi]
        else:
            logger.warning("Don't know how to update local state!")
        self._local[2] = clip_angle(self._local[2])

    def update_action(self, action):
        if self._local is not None:
            self._local = self._f(list(self._local), (action[0], -action[1]))
            logger.debug(
                "Update local state to %s after action %s", self._local, action
            )

    def update_plan(self, node_kind, command):
        self._current_node_kind = node_kind
        self._current_command = command

    def update_perception(self, perception_output):
        # logger.debug("Update perception error at a %s tile", self._current_node_kind)
        logger.debug("Update perception error at a %s tile", self._current_node_kind)
        (
            white_line,
            yellow_line,
            junction_features,
            stopline_features,
        ) = perception_output
        if self._current_node_kind is None and self._local is None:
            if stopline_features is not None:
                self._stop_line_init_count += 1
            else:
                self._stop_line_init_count = 0
            if self._stop_line_init_count > 3 and stopline_features is not None:
                self._process_init_features(stopline_features)
        if self._current_node_kind is None or self._current_node_kind == "straight":
            if white_line is not None:
                self._process_straight_white_line(white_line)
            if yellow_line is not None and self._current_node_kind == 'straight':
                self._process_yellow_line_straight(yellow_line)
            # elif stopline_features is not None:
            #     self._process_straight_stopline(stopline_features)
        if self._current_node_kind is not None:
            self._process_both_lane_lines(white_line, yellow_line)
        if self._current_node_kind == "junction":
            if stopline_features is not None:
                self._process_junction_features(stopline_features)
            if self._current_command == 'forward':
                self._process_straight_white_line(white_line)
        if self._current_node_kind == "lturn":
            if white_line is not None:
                self._process_lturn_white_line(white_line)
        if self._current_node_kind == "rturn":
            if white_line is not None:
                self._process_rturn_white_line(white_line)

    def _process_init_features(self, stopline_features):
        next_stopline_gt = np.array(
            [1 + self._red_centroid_positions[1], 1 - self._red_centroid_positions[0]]
        )

        stopline_centroid, stopline_dir = stopline_features
        distance_to_centroid = np.linalg.norm(stopline_centroid.flatten())
        if distance_to_centroid < 0.3 or distance_to_centroid > 1:
            logger.info("Initial stopline not in range: %.3f", distance_to_centroid)
            return
        logger.info("Initial stopline centroid: %s", stopline_centroid)
        logger.info("Initial stopline direction: %s", stopline_dir)
        # z-axis is pointing downwards!
        cos_angle = np.arccos(abs(stopline_dir[0]))
        logger.info("Iniital stopline angle: %s", cos_angle)
        unsigned_angle = np.pi / 2 - cos_angle
        current_angle = -unsigned_angle if stopline_dir[1] * stopline_dir[0] < 0 else unsigned_angle

        R = np.array(
            [
                [np.cos(current_angle), -np.sin(current_angle)],
                [np.sin(current_angle), np.cos(current_angle)],
            ]
        )
        aligned_stopline_centroid = np.squeeze(np.matmul(R, stopline_centroid))
        current_position = next_stopline_gt - aligned_stopline_centroid

        # Update the state
        self._local = [current_position[0], current_position[1], current_angle]
        logger.info("Initialising state: %s, %s", current_position, current_angle)

    def _process_straight_white_line(self, white_line):
        logger.debug("Attemp to estimate state from straight white line")
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            white_line[0], white_line[1]
        )
        logger.debug("White line linear regression err: %s", std_err)
        estimated_angle = -np.arctan(slope)  # angle of the robot in tile frame
        estimated_distance = intercept / np.sqrt(slope * slope + 1)

        if -0.1 < estimated_angle < 0.1 and 0.2 < estimated_distance < 0.4:
            self.can_stop = True

        dist_thresh = 0.8
        if abs(estimated_distance) > dist_thresh:
            logger.debug("Too far away: %.2f > %.2f", estimated_distance, dist_thresh)
            return
        std_err_thresh = 0.02
        if std_err > std_err_thresh:
            logger.debug("Std_err too large: %.2f > %.2f", std_err, std_err_thresh)
            return
        length = np.linalg.norm(white_line[:, 0] - white_line[:, -1])
        length_thresh = 0.4
        if length < length_thresh:
            logger.debug("Too short: %.2f < %.2f", length, length_thresh)
            return
        estimated_y = (
            (1 - estimated_distance)
            if estimated_distance > 0
            else 0.12 - estimated_distance
        )
        logger.debug(
            "White line slope (y = kx + b): k=%s, b=%s. Distance: %s, angle: %s",
            slope,
            intercept,
            estimated_distance,
            estimated_angle,
        )
        if self._local is None:
            self._local = [0, estimated_y, estimated_angle]
        else:
            # TODO: Change to a proper low pass filter
            self._local[1] = 0.5 * estimated_y + 0.5 * self._local[1]
            self._local[2] = 0.5 * estimated_angle + 0.5 * self._local[2]
        logger.debug("Update local state to %s", self._local)

    def _process_lturn_white_line(self, white_line):
        logger.debug("Attemp to estimate state from left turn")
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            white_line[0], white_line[1]
        )
        logger.debug("White line linear regression err: %s", std_err)
        estimated_angle = -np.pi / 2 - np.arctan(
            slope
        )  # angle of the robot in tile frame
        estimated_distance = intercept / np.sqrt(slope * slope + 1)
        dist_thresh = 0.8
        if estimated_distance < 0:
            logger.debug("Distance negative (left hand side). Wrong side.")
            return
        if estimated_distance > dist_thresh:
            logger.debug("Too far away: %.2f > %.2f", estimated_distance, dist_thresh)
            return
        std_err_thresh = 0.02
        if std_err > std_err_thresh:
            logger.debug("Std_err too large: %.2f > %.2f", std_err, std_err_thresh)
            return
        if abs(np.arctan(slope)) > np.pi / 6:
            logger.debug("Std_err too large: %.2f > %.2f")
            return
        length = np.linalg.norm(white_line[:, 0] - white_line[:, -1])
        length_thresh = 0.5
        if length < length_thresh:
            logger.debug("Too short: %.2f < %.2f", length, length_thresh)
            return
        estimated_x = 1 - estimated_distance
        logger.debug(
            "White line slope (y = kx + b): k=%s, b=%s. Distance: %s, angle: %s",
            slope,
            intercept,
            estimated_distance,
            estimated_angle,
        )
        # TODO: Change to a proper low pass filter
        self._local[0] = 0.5 * estimated_x + 0.5 * self._local[0]
        self._local[2] = 0.5 * estimated_angle + 0.5 * self._local[2]
        logger.debug("Update local state to %s", self._local)

    def _process_rturn_white_line(self, white_line):
        logger.debug("Attemp to estimate state from right turn")
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            white_line[0], white_line[1]
        )
        logger.debug("White line linear regression err: %s", std_err)
        estimated_angle = np.pi / 2 - np.arctan(
            slope
        )  # angle of the robot in tile frame
        estimated_distance = intercept / np.sqrt(slope * slope + 1)
        dist_thresh = 0.8
        if estimated_distance > 0:
            logger.debug("Distance positive (right hand side). Wrong side.")
            return
        estimated_distance = 0.12 - estimated_distance
        if estimated_distance > dist_thresh:
            logger.debug("Too far away: %.2f > %.2f", estimated_distance, dist_thresh)
            return
        std_err_thresh = 0.02
        if std_err > std_err_thresh:
            logger.debug("Std_err too large: %.2f > %.2f", std_err, std_err_thresh)
            return
        length = np.linalg.norm(white_line[:, 0] - white_line[:, -1])
        length_thresh = 0.7
        if length < length_thresh:
            logger.debug("Too short: %.2f < %.2f", length, length_thresh)
            return
        estimated_x = 1 - estimated_distance
        logger.debug(
            "White line slope (y = kx + b): k=%s, b=%s. Distance: %s, angle: %s",
            slope,
            intercept,
            estimated_distance,
            estimated_angle,
        )
        # TODO: Change to a proper low pass filter
        self._local[0] = 0.5 * estimated_x + 0.5 * self._local[0]
        self._local[2] = 0.5 * estimated_angle + 0.5 * self._local[2]
        logger.debug("Update local state to %s", self._local)

    def _process_junction_features(self, stopline_features):
        local_stopline_gt = np.array([0.95, 0.25])
        stopline_centroid, _ = stopline_features
        current_x, current_y, current_angle = self._local

        R = np.array(
            [
                [np.cos(current_angle), -np.sin(current_angle)],
                [np.sin(current_angle), np.cos(current_angle)],
            ]
        )
        aligned_stopline_centroid = np.squeeze(np.matmul(R, stopline_centroid))

        # We only want to use stoplines that are nearby (not too far away else they might be in another tile) and
        # we want to make sure that in a junction we're only using the top-left stopline to localize against
        current_position = np.array([current_x, current_y])
        if (
            np.linalg.norm(aligned_stopline_centroid) < 0.7
            and np.linalg.norm(
                current_position + aligned_stopline_centroid - local_stopline_gt
            )
            < 0.3
        ):
            estimated_position = local_stopline_gt - aligned_stopline_centroid
            estimated_x, _ = estimated_position
            self._local[0] = 0.5 * estimated_x + 0.5 * self._local[0]
            logger.info("Updating state using JUNCTION features: %s", estimated_x)

    def _process_straight_stopline(self, stopline_features):
        local_stopline_gt = np.array([1.05, 0.75])
        stopline_centroid, _ = stopline_features
        current_x, current_y, current_angle = self._local

        R = np.array(
            [
                [np.cos(current_angle), -np.sin(current_angle)],
                [np.sin(current_angle), np.cos(current_angle)],
            ]
        )
        aligned_stopline_centroid = np.squeeze(np.matmul(R, stopline_centroid))

        # We only want to use stoplines that are nearby (not too far away else they might be in another tile) and
        # we want to make sure that in a junction we're only using the top-left stopline to localize against
        current_position = np.array([current_x, current_y])
        if (
            np.linalg.norm(aligned_stopline_centroid) < 1.0
            and np.linalg.norm(
                current_position + aligned_stopline_centroid - local_stopline_gt
            )
            < 0.3
        ):
            estimated_position = local_stopline_gt - aligned_stopline_centroid
            estimated_x, _ = estimated_position
            self._local[0] = 0.5 * estimated_x + 0.5 * self._local[0]
            logger.debug("Updating state using JUNCTION features: %s", estimated_x)

    def _process_yellow_line_straight(self, yellow_line):
        logger.debug("Attemp to estimate state from straight yellow line %s", yellow_line)
        R_tile_robot = np.array(
            [
                [np.cos(self._local[2]), -np.sin(self._local[2])],
                [np.sin(self._local[2]), np.cos(self._local[2])],
            ]
        )
        for (x, y) in zip(yellow_line[0], yellow_line[1]):
            if y < 0 and y > -0.1 and x < 0.3:
                logger.debug("Too close to yellow line. Correct local state.")
                update_vec = R_tile_robot @ np.array([0, -1])
                self._local[0] = self._local[0] + 0.02 * update_vec[1]
                self._local[1] = self._local[1] + 0.02 * update_vec[1]
                break

    def _process_both_lane_lines(self, white_line, yellow_line):
        """Correct robot local state if white/yellow border is too close."""
        logger.debug("Attemp to estimate state from straight yellow line %s", yellow_line)
        R_tile_robot = np.array(
            [
                [np.cos(self._local[2]), -np.sin(self._local[2])],
                [np.sin(self._local[2]), np.cos(self._local[2])],
            ]
        )
        if white_line is not None:
            for (x, y) in zip(white_line[0], white_line[1]):
                if y > 0 and y < 0.20 and x < 0.3:
                    logger.debug("Too close to right white line. Correct local state.")
                    update_vec = R_tile_robot @ np.array([0, 1])
                    self._local[0] = self._local[0] + 0.01 * update_vec[1]
                    self._local[1] = self._local[1] + 0.01 * update_vec[1]
                    logger.debug("Corrected local state: %s", self._local)
                    break
        if yellow_line is not None:
            for (x, y) in zip(yellow_line[0], yellow_line[1]):
                if y < 0 and y > -0.1 and x < 0.15:
                    logger.debug("Too close to left yellow line. Correct local state.")
                    update_vec = R_tile_robot @ np.array([0, -1])
                    self._local[0] = self._local[0] + 0.02 * update_vec[1]
                    self._local[1] = self._local[1] + 0.02 * update_vec[1]
                    break
