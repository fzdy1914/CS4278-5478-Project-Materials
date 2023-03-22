import json
import time

import numpy as np
import ray
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper


f = open("./testcases/milestone1.json", "r")
task_dict = json.load(f)
for map_name, task_info in task_dict.items():
    # if map_name != "map3_4":
    #     continue

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    env = DuckietownEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=map_name,
        seed=seed,
        user_tile_start=start_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False,
        my_mode="none"
    )

    print(map_name, env.cur_pos, env.cur_angle)
