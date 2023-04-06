import math
import random

import numpy as np
from gymnasium import spaces
from gym_duckietown import logger
from gymnasium.wrappers.compatibility import LegacyEnv

from . import DuckietownEnv

DIR_TO_NUM = {
    "forward_first(not used)": 0,
    "left": 1,
    "right:": 2,
    "forward_normal": 3,
    "goal": 4,
}

uncertainty = 0


def identity(x):
    return x


def ceil(x):
    return math.ceil(x) - 0.0001


def floor(x):
    return math.floor(x) + 0.0001


def new_ceil(x):
    up = min(0.295, uncertainty)
    down = max(-0.695, -uncertainty)
    pos = random.uniform(down, up)

    return math.ceil(x) - 0.3 + pos


def new_floor(x):
    up = min(0.295, uncertainty)
    down = max(-0.695, -uncertainty)
    pos = random.uniform(down, up)

    return math.floor(x) + 0.3 - pos


def goal_ceil(x):
    return math.ceil(x) - 0.3


def goal_floor(x):
    return math.floor(x) + 0.3


class DirectedBotEnv(DuckietownEnv):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(
        self,
        direction=0,
        **kwargs
    ):
        self.next_locations = {"map1_0": [1, 1], "map1_1": [1, 1], "map1_2": [3, 1], "map1_3": [6, 1], "map1_4": [51, 1], "map2_0": [7, 6], "map2_1": [1, 2], "map2_2": [7, 6], "map2_3": [3, 1], "map2_4": [1, 5], "map3_0": [5, 1], "map3_1": [2, 2], "map3_2": [7, 4], "map3_3": [10, 7], "map3_4": [5, 2], "map4_0": [4, 3], "map4_1": [1, 7], "map4_2": [12, 11], "map4_3": [13, 7], "map4_4": [12, 4], "map5_0": [11, 4], "map5_1": [5, 13], "map5_2": [13, 4], "map5_3": [12, 14], "map5_4": [2, 3]}
        self.goal_obj_position = {
            "map1_0": [5.5, 0.9, 0.2],
            "map1_1": [70, 0.9, 0.2],
            "map1_2": [21.1, 2.01, 0.2],
            "map1_3": [65.5, 2.02, 0.2],
            "map1_4": [90.8, 2.01, 0.2],
            "map2_0": [0.95, 1.5, 0.2],
            "map2_1": [7.5, 0.9, 0.2],
            "map2_2": [2.9, 4.5, 0.2],
            "map2_3": [6.1, 4.5, 0.2],
            "map2_4": [4.5, 6.9, 0.2],
            "map3_0": [1.9, 2.5, 0.2],
            "map3_1": [0.9, 7.3, 0.2],
            "map3_2": [7.5, 10.9, 0.2],
            "map3_3": [9.5, 2.1, 0.2],
            "map3_4": [11.1, 11.5, 0.2],
            "map4_0": [2.9, 3.5, 0.2],
            "map4_1": [0.9, 12.4, 0.2],
            "map4_2": [11.73, 12.13, 0.2],
            "map4_3": [14.2, 8.7, 0.2],
            "map4_4": [11.5, 3.9, 0.2],
            "map5_0": [1.9, 9.5, 0.2],
            "map5_1": [4.5, 14.1, 0.2],
            "map5_2": [9.9, 1.4, 0.2],
            "map5_3": [13.1, 15.33, 0.2],
            "map5_4": [14.9, 9.46, 0.2]
        }

        self.actions = {
            (1, 0): (math.floor, goal_ceil, 0),
            (0, -1): (goal_ceil, math.ceil, 0.5 * np.pi),
            (-1, 0): (math.ceil, goal_floor, np.pi),
            (0, 1): (goal_floor, math.floor, 1.5 * np.pi)
        }
        self.direction = direction
        my_mode = "cross"
        if self.direction == 3:
            my_mode = "none"
        if self.direction == 0:
            my_mode = "start"
        if self.direction == 4:
            my_mode = "goal"

        DuckietownEnv.__init__(self, my_mode=my_mode, **kwargs)
        logger.info('using DirectedBotEnv')
        if self.direction == 2:
            self.action_space = spaces.Box(
                low=np.array([.25, -np.pi]),
                high=np.array([1, 0]),
                dtype=np.float64
            )
        elif self.direction == 1:
            self.action_space = spaces.Box(
                low=np.array([.25, -0.5 * np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )
        elif direction == 0:
            self.action_space = spaces.Box(
                low=np.array([-0.25, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )
        elif direction == 3:
            self.action_space = spaces.Box(
                low=np.array([0.25, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )
        elif direction == 4:
            self.action_space = spaces.Box(
                low=np.array([-0.25, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float64
            )

        self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            )

    def generate_goal_tile_forward_first(self):
        self.start_location = self.get_grid_coords(self.cur_pos)
        next_location = self.next_locations[self.map_name]

        action = (next_location[0] - self.start_location[0], next_location[1] - self.start_location[1])

        ideal_op_x, ideal_op_y, ideal_angle = self.actions[action]

        new_pos_x = self.start_location[0] + action[0]
        new_pos_y = self.start_location[1] + action[1]

        for tile in self.drivable_tiles:
            if tile['coords'] == (new_pos_x, new_pos_y):
                self.goal_location = (new_pos_x, new_pos_y)
                self.goal_pos = (ideal_op_x(self.cur_pos[0] + action[0]), ideal_op_y(self.cur_pos[2] + action[1]))
                self.ideal_angle = ideal_angle
                return True

        print("b")
        return False

    def generate_goal_tile_left(self):
        start_location = self.get_grid_coords(self.cur_pos)
        self.start_location = start_location
        tile = self._get_tile(start_location[0], start_location[1])
        angle = tile["angle"]
        kind = tile['kind']

        if "straight" in kind:
            return False

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
            ideal_op_x = goal_ceil
            ideal_op_y = math.ceil
            ideal_angle = 0.5 * np.pi

            op_x = floor
            op_y = new_ceil
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 2:
                return False

            action = (-1, 0)
            ideal_op_x = math.ceil
            ideal_op_y = goal_floor
            ideal_angle = np.pi

            op_x = new_ceil
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            if kind == 'curve_right' and angle != 2:
                return False
            if kind == 'curve_left' and angle != 3:
                return False

            action = (0, 1)
            ideal_op_x = goal_floor
            ideal_op_y = math.floor
            ideal_angle = 1.5 * np.pi

            op_x = ceil
            op_y = new_floor
        else:
            if kind == 'curve_right' and angle != 3:
                return False
            if kind == 'curve_left' and angle != 0:
                return False
            action = (1, 0)

            ideal_op_x = math.floor
            ideal_op_y = goal_ceil
            ideal_angle = 0

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
                self.ideal_angle = ideal_angle
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
            ideal_op_x = goal_floor
            ideal_op_y = math.floor
            ideal_angle = 1.5 * np.pi

            op_x = floor
            op_y = new_ceil
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            if kind == 'curve_right' and angle != 2:
                return False
            if kind == 'curve_left' and angle != 3:
                return False

            action = (1, 0)
            ideal_op_x = math.floor
            ideal_op_y = goal_ceil
            ideal_angle = 0

            op_x = new_ceil
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            if kind == 'curve_right' and angle != 3:
                return False
            if kind == 'curve_left' and angle != 0:
                return False

            action = (0, -1)
            ideal_op_x = goal_ceil
            ideal_op_y = math.ceil
            ideal_angle = 0.5 * np.pi

            op_x = ceil
            op_y = new_floor
        else:
            if kind == 'curve_right':
                return False
            if kind == 'curve_left' and angle != 1:
                return False
            action = (-1, 0)
            ideal_op_x = math.ceil
            ideal_op_y = goal_floor
            ideal_angle = np.pi

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
                self.ideal_angle = ideal_angle
                return True

        return False

    def generate_goal_tile_forward_normal(self):
        start_location = self.get_grid_coords(self.cur_pos)
        self.start_location = start_location

        if math.fabs(self.cur_angle) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 1 / 2 * np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - np.pi) > 1 / 8 * np.pi and \
                math.fabs(self.cur_angle - 3 / 2 * np.pi) > 1 / 8 * np.pi:
            return False

        if self.cur_angle > 7 / 4 * np.pi or self.cur_angle <= 1 / 4 * np.pi:
            op_x = floor
            op_y = new_ceil
        elif 1 / 4 * np.pi < self.cur_angle <= 3 / 4 * np.pi:
            op_x = new_ceil
            op_y = ceil
        elif 3 / 4 * np.pi < self.cur_angle <= 5 / 4 * np.pi:
            op_x = ceil
            op_y = new_floor
        else:
            op_x = new_floor
            op_y = floor
        self.cur_pos[0] = op_x(self.cur_pos[0])
        self.cur_pos[2] = op_y(self.cur_pos[2])
        return True

    def reset(self):
        obs = DuckietownEnv.reset(self)

        self.randomize_maps_on_reset = False

        if self.direction == 0:
            if not self.generate_goal_tile_forward_first():
                return self.reset()
        elif self.direction == 1:
            if not self.generate_goal_tile_left():
                return self.reset()
        elif self.direction == 2:
            if not self.generate_goal_tile_right():
                return self.reset()
        elif self.direction == 3:
            if not self.generate_goal_tile_forward_normal():
                return self.reset()
        elif self.direction == 4:
            if not self.generate_goal_tile_forward_normal():
                return self.reset()
        elif self.direction == 5:
            return obs

        if not self.valid_pose(self.cur_pos, self.cur_angle):
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
            if self.direction != 4:
                if self.get_grid_coords(self.cur_pos) == self.goal_location:
                    reward = 100
                    dist = math.sqrt((self.cur_pos[0] - self.goal_pos[0]) ** 2 + (self.cur_pos[2] - self.goal_pos[1]) ** 2)
                    angle_diff = min(math.fabs((self.cur_angle % (2 * np.pi)) - self.ideal_angle),
                                     math.fabs((self.cur_angle % (2 * np.pi)) - 2 * np.pi - self.ideal_angle),
                                     math.fabs((self.cur_angle % (2 * np.pi)) + 2 * np.pi - self.ideal_angle))
                    reward -= 100 * dist + 25 * angle_diff
                else:
                    reward = -100
                done = True
            else:
                reward = -100
                done = True

        if self.direction == 4:
            location = self.goal_obj_position[self.map_name]
            dist = math.sqrt((location[0] - self.cur_pos[0]) ** 2 + (location[1] - self.cur_pos[2]) ** 2)
            print(dist)
            reward += 10 * (1 - dist)
        else:
            if not done:
                reward -= 1

        return obs, reward, done, info
