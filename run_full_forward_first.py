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

delta_to_direction = {
    (1, 0): 0,
    (0, -1): 1,
    (-1, 0): 2,
    (0, 1): 3,
}


def launch_and_wrap_env(ctx):
    env = DirectedBotEnv(direction=ctx["direction"], domain_rand=False, max_steps=1500, map_name="map1_0", randomize_maps_on_reset=True)

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env


ray.init()
register_env("MyDuckietown", launch_and_wrap_env)

config = PPOConfig().environment("MyDuckietown", env_config={"direction": 0}).framework("torch").rollouts(num_rollout_workers=0).resources(num_gpus=0)
algo_forward_first = config.build()
algo_forward_first.restore("D:\\forward_result\\checkpoint_000968")

right_action = [[-0.8, np.pi]] * 15 + [[1, 0]] * 12
left_action = [[-0.9, -np.pi]] * 25

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

for map_name, task_info in task_dict.items():
    # if "map4_0" not in map_name:
    #     continue

    actions = []
    total_reward = 0
    total_step = 0

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    intentions = {start_tile: "forward"}

    print(map_name, seed, start_tile, goal_tile)

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

    env = EnvCompatibility(env_old)
    env = ResizeWrapper(env)
    env_stack = StackWrapper(env)
    env = NormalizeWrapper(env_stack)

    map_img, goal, start_pos = env_old.get_task_info()

    # start first tile handling
    robot = LaneFollower(intentions, map_img, goal, visualize=False)

    action = [0, 0]
    obs, _, _, _,  info = env.step(action)
    env_old.render()

    mode = "Follower"
    while info["curr_pos"] == start_tile:
        action = algo_forward_first.compute_single_action(
            observation=obs,
            explore=False,
        )
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_step += 1
        actions.append(action)
        env_old.render()




