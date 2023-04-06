import json
import time

import numpy as np
import ray
from gymnasium.wrappers import EnvCompatibility
from intelligent_robots_project import LaneFollower
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from path_generating import generate_path

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

config = PPOConfig().environment("MyDuckietown", env_config={"direction": 3}).framework("torch").rollouts(num_rollout_workers=0).resources(num_gpus=0)
algo_forward_normal = config.build()
algo_forward_normal.restore("./forward_normal_result/final_best")
#
# config = (
#         PPOConfig()
#         .environment("MyDuckietown", env_config={
#             "direction": 1
#         })
#         .framework("torch")
#         .rollouts(num_rollout_workers=0)
#         .resources(num_gpus=0)
#     )
# algo_left = config.build()
# algo_left.restore("./left_result/final_best")
#
# config = (
#         PPOConfig()
#         .environment("MyDuckietown", env_config={
#             "direction": 2
#         })
#         .framework("torch")
#         .rollouts(num_rollout_workers=0)
#         .resources(num_gpus=0)
#     )
# algo_right = config.build()
# algo_right.restore("./right_result/final_best")
#
# algos = {
#     "forward": algo_forward_normal,
#     "left": algo_left,
#     "right": algo_right,
# }

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

for map_name, task_info in task_dict.items():
    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    tiles = [start_tile]
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
    robot = LaneFollower(intentions, map_img, goal, visualize=True)

    action = [0, 0]
    obs, _, _, info = env_old.step(action)
    # env_old.render()

    mode = "Follower"
    while info["curr_pos"] == start_tile:
        if mode == "Follower":
            action = robot(obs, info, action)
            if robot.state_estimator.can_stop:
                mode = "RL"
                obs, _, _, _, info = env.step([0, 0])
                continue
            obs, reward, done, info = env_old.step(action)
            # env_old.render()
        else:
            action = algo_forward_normal.compute_single_action(
                observation=obs,
                explore=False,
            )
            obs, reward, done, truncated, info = env.step(action)
            # env_old.render()

    delta = (info["curr_pos"][0] - start_tile[0], info["curr_pos"][1] - start_tile[1])
    direction = delta_to_direction[delta]
    print(start_tile, info["curr_pos"], delta, direction)

    # forward, backward, left, right
    control_path = generate_path(map_img, start_pos, goal, direction)
    print(control_path)
