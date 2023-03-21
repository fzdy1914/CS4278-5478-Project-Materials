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


def identity(x):
    return x


def ceil(x):
    return math.ceil(x) - 0.1


def floor(x):
    return math.floor(x) + 0.1


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
        self.direction=direction
        LegacyEnv.__init__(self)
        DuckietownEnv.__init__(self, **kwargs)
        logger.info('using GuidedBotEnvEnv')
        self.action_space = spaces.Box(
            low=np.array([.25,-np.pi]),
            high=np.array([1,np.pi]),
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

        if math.fabs(self.cur_angle) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 1 / 2 * np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 3 / 2 * np.pi) > 1 / 8 * np.pi:
            return False

        if self.cur_angle > 7 / 4 * np.pi or self.cur_angle <= 1 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 1:
                return False

            action = (0, -1)
            op_x = floor
            op_y = identity
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 2:
                return False

            action = (-1, 0)
            op_x = identity
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            if kind == 'curve_right' and angle != 2:
                return False
            if kind == 'curve_left' and angle != 3:
                return False

            action = (0, 1)
            op_x = ceil
            op_y = identity
        else:
            if kind == 'curve_right' and angle != 3:
                return False
            if kind == 'curve_left' and angle != 0:
                return False
            action = (1, 0)
            op_x = identity
            op_y = floor

        new_pos_x = start_location[0] + action[0]
        new_pos_y = start_location[1] + action[1]
        for tile in self.drivable_tiles:
            if tile['coords'] == (new_pos_x, new_pos_y):
                self.goal_location = (new_pos_x, new_pos_y)
                self.cur_pos[0] = op_x(self.cur_pos[0])
                self.cur_pos[2] = op_y(self.cur_pos[2])
                return True

        return False

    def reset(self):
        obs = DuckietownEnv.reset(self)
        if not self.generate_goal_tile_left():
            return self.reset()

        return obs

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)

        if reward < -100:
            reward = -100

        if self.get_grid_coords(self.cur_pos) != self.start_location:
            if self.get_grid_coords(self.cur_pos) == self.goal_location:
                reward = 100
            else:
                reward = -100
            done = True

        if not done:
            reward -= 1

        return obs, reward, done, info
