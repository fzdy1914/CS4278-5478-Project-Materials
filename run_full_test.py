import json
import math
import time

import numpy as np
import ray
import torch
from gymnasium.wrappers import EnvCompatibility

from cnn_model import RegressionResNet
from find_highest_peak import find_highest_peak
from intelligent_robots_project import LaneFollower
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from torchvision import models, transforms

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from path_generating import generate_path
from gym_duckietown.envs.directed_bot_env import goal_obj_position

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

for map_name, task_info in task_dict.items():
    if "map4_4" != map_name:
        continue

    total_reward = 0
    total_step = 0

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])
    print(map_name, seed, start_tile, goal_tile)

    actions = np.loadtxt(
        f'./m2_control_files/{map_name}_seed{seed}_start_{start_tile[0]},{start_tile[1]}_goal_{goal_tile[0]},{goal_tile[1]}.txt',
        delimiter=',')

    env_old = DuckietownEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=map_name,
        seed=seed,
        user_tile_start=start_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False,
        my_mode="none",
    )

    env_old.render()
    for action in actions:
        _, reward, done, _ = env_old.step(action)
        total_step += 1
        total_reward += reward
        env_old.render()

    location = goal_obj_position[env_old.map_name]
    dist = math.sqrt((location[0] - env_old.cur_pos[0]) ** 2 + (location[1] - env_old.cur_pos[2]) ** 2)
    print(total_reward, total_step, total_reward / total_step, dist)
