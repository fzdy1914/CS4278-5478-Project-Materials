import math
import random

import numpy as np
from gymnasium import spaces
from gym_duckietown import logger
from gymnasium.wrappers.compatibility import LegacyEnv

from . import DuckietownEnv

DIR_TO_NUM = {
    "forward": 0,
    "left": 1,
    "right:": 2,
}

uncertainty = 0.15


def identity(x):
    return x


def ceil(x):
    return math.ceil(x) - random.random() * uncertainty


def floor(x):
    return math.floor(x) + random.random() * uncertainty


def new_ceil(x):
    return math.ceil(x) - 0.25 + (random.random() - 0.5) * uncertainty


def new_floor(x):
    return math.floor(x) + 0.25 + (random.random() - 0.5) * uncertainty


class DirectedBotEnv(DuckietownEnv, LegacyEnv):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(
        self,
        direction=0,
        **kwargs
    ):
        self.direction = direction
        LegacyEnv.__init__(self)
        DuckietownEnv.__init__(self, **kwargs)
        logger.info('using GuidedBotEnvEnv')
        if self.direction == 2:
            self.action_space = spaces.Box(
                low=np.array([.25, -np.pi]),
                high=np.array([1, 0]),
                dtype=np.float64
            )
        elif self.direction == 1:
            self.action_space = spaces.Box(
                low=np.array([.25, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([.25, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )

        self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            )

    def generate_goal_tile_left(self):
        start_location = self.get_grid_coords(self.cur_pos)
        self.start_location = start_location
        tile = self._get_tile(start_location[0], start_location[1])
        angle = tile["angle"]
        kind = tile['kind']

        if "straight" in kind:
            return False

        if self.direction != 0:
            if math.fabs(self.cur_angle) > 1 / 6 * np.pi and \
                    math.fabs(self.cur_angle - 1 / 2 * np.pi) > 1 / 6 * np.pi and \
                    math.fabs(self.cur_angle - np.pi) > 1 / 6 * np.pi and \
                    math.fabs(self.cur_angle - 3 / 2 * np.pi) > 1 / 6 * np.pi:
                return False

        if self.cur_angle > 7 / 4 * np.pi or self.cur_angle <= 1 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 1:
                return False

            action = (0, -1)
            ideal_op_x = new_ceil
            ideal_op_y = math.ceil

            op_x = floor
            op_y = new_ceil
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 2:
                return False

            action = (-1, 0)
            ideal_op_x = math.ceil
            ideal_op_y = new_floor

            op_x = new_ceil
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            if kind == 'curve_right' and angle != 2:
                return False
            if kind == 'curve_left' and angle != 3:
                return False

            action = (0, 1)
            ideal_op_x = new_floor
            ideal_op_y = math.floor

            op_x = ceil
            op_y = new_floor
        else:
            if kind == 'curve_right' and angle != 3:
                return False
            if kind == 'curve_left' and angle != 0:
                return False
            action = (1, 0)

            ideal_op_x = math.floor
            ideal_op_y = new_ceil
            op_x = new_floor
            op_y = floor

        new_pos_x = start_location[0] + action[0]
        new_pos_y = start_location[1] + action[1]
        for tile in self.drivable_tiles:
            if tile['coords'] == (new_pos_x, new_pos_y):
                self.goal_location = (new_pos_x, new_pos_y)
                self.cur_pos[0] = op_x(self.cur_pos[0])
                self.cur_pos[2] = op_y(self.cur_pos[2])
                self.goal_pos = (ideal_op_x(self.cur_pos[0] + action[0]), ideal_op_y(self.cur_pos[2] + action[1]))
                return True

        return False

    def generate_goal_tile_right(self):
        start_location = self.get_grid_coords(self.cur_pos)
        self.start_location = start_location
        tile = self._get_tile(start_location[0], start_location[1])
        angle = tile["angle"]
        kind = tile['kind']

        if "straight" in kind:
            return False

        if math.fabs(self.cur_angle) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 1 / 2 * np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 3 / 2 * np.pi) > 1 / 8 * np.pi:
            return False

        if self.cur_angle > 7 / 4 * np.pi or self.cur_angle <= 1 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 2:
                return False

            action = (0, 1)
            ideal_op_x = new_floor
            ideal_op_y = math.floor

            op_x = floor
            op_y = new_ceil
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            if kind == 'curve_right' and angle != 2:
                return False
            if kind == 'curve_left' and angle != 3:
                return False

            action = (1, 0)
            ideal_op_x = math.floor
            ideal_op_y = new_ceil

            op_x = new_ceil
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            if kind == 'curve_right' and angle != 3:
                return False
            if kind == 'curve_left' and angle != 0:
                return False

            action = (0, -1)
            ideal_op_x = new_ceil
            ideal_op_y = math.ceil

            op_x = ceil
            op_y = new_floor
        else:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 1:
                return False
            action = (-1, 0)
            ideal_op_x = math.ceil
            ideal_op_y = new_floor

            op_x = new_floor
            op_y = floor

        new_pos_x = start_location[0] + action[0]
        new_pos_y = start_location[1] + action[1]
        for tile in self.drivable_tiles:
            if tile['coords'] == (new_pos_x, new_pos_y):
                self.goal_location = (new_pos_x, new_pos_y)
                self.cur_pos[0] = op_x(self.cur_pos[0])
                self.cur_pos[2] = op_y(self.cur_pos[2])
                self.goal_pos = (ideal_op_x(self.cur_pos[0] + action[0]), ideal_op_y(self.cur_pos[2] + action[1]))
                return True

        return False

    def reset(self):
        obs = DuckietownEnv.reset(self)

        self.randomize_maps_on_reset = False

        if self.direction == 1:
            if not self.generate_goal_tile_left():
                return self.reset()
        elif self.direction == 2:
            if not self.generate_goal_tile_right():
                return self.reset()

        if not self._valid_pose(self.cur_pos, self.cur_angle):
            return self.reset()

        _, _, done, info = self.step([0, 0])
        if done:
            return self.reset()

        self.randomize_maps_on_reset = True
        return obs

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)

        if reward < -100:
            reward = -100

        if self.get_grid_coords(self.cur_pos) != self.start_location:
            if self.get_grid_coords(self.cur_pos) == self.goal_location:
                reward = 100
                dist = math.sqrt((self.cur_pos[0] - self.goal_pos[0]) ** 2 + (self.cur_pos[2] - self.goal_pos[1]) ** 2)
                reward -= 100 * dist
            else:
                reward = -100
            done = True

        if not done:
            reward -= 1

        return obs, reward, done, info
