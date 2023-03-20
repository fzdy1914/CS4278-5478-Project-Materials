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
        self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            )


    def generate_goal_tile(self):
        self.start_location = self.get_grid_coords(self.cur_pos)

        if self.cur_angle > 7 / 4 * np.pi or self.cur_angle <= 1 / 4 * np.pi:
            forward = (1, 0)
            left = (0, -1)
            right = (0, 1)
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            forward = (0, -1)
            left = (-1, 0)
            right = (1, 0)
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            forward = (-1, 0)
            left = (0, 1)
            right = (0, -1)
        else:
            forward = (0, 1)
            left = (1, 0)
            right = (-1, 0)

        can = False
        for guide, (x, y) in [(0, forward), (1, left), (2, right)]:
            if guide != self.direction:
                continue

            new_pos_x = self.start_location[0] + x
            new_pos_y = self.start_location[1] + y
            for tile in self.drivable_tiles:
                if tile['coords'] == (new_pos_x, new_pos_y):
                    self.goal_location = (new_pos_x, new_pos_y)
                    can = True
                    break

        if not can:
            return False

        return True

    def reset(self):
        obs = DuckietownEnv.reset(self)
        if not self.generate_goal_tile():
            return self.reset()

        return obs

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)

        if reward < -10:
            reward = -10

        if self.get_grid_coords(self.cur_pos) != self.start_location:
            if self.get_grid_coords(self.cur_pos) == self.goal_location:
                reward = 10
            else:
                reward = -10
            done = True

        if not done:
            reward -= 1

        return obs, reward, done, info
