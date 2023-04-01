import json
import time

import numpy as np
import ray
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper


def launch_and_wrap_env(ctx):
    env = DirectedBotEnv(
        direction=ctx["direction"],
        domain_rand=False,
        max_steps=1500,
        map_name="map1_0",
        randomize_maps_on_reset=True
    )

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env


ray.init()
register_env('MyDuckietown', launch_and_wrap_env)

config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 3
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )
algo_forward_normal = config.build()
algo_forward_normal.restore("D:\\forward_normal_result\\checkpoint_000542")

config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 1
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )
algo_left = config.build()
algo_left.restore("D:\\left_result\\checkpoint_000700")

config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 2
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )
algo_right = config.build()
algo_right.restore("./right_result/final_best")

algos = {
    "forward": algo_forward_normal,
    "left": algo_left,
    "right": algo_right,
}

dir_path = "./testcases/milestone1_paths/"

f = open("./testcases/milestone1.json", "r")
task_dict = json.load(f)

# this looks like a hack, but we believe solving it belongs the scope of milestone 2
# we will rotate and find one available tile to go in order to know the angle of the robot.
# The route planning will then based on the new angle.
# However, in milestone 1, the route is fixed, thus, we have to do this to avoid the issue of bad spawning.
predefined_action_list = {
    "map1_1": [[0, np.pi]] * 5,
    "map1_2": [[0, -np.pi]] * 5,
    "map1_4": [[0, np.pi]] * 5,
    "map2_4": [[0, np.pi]] * 6,
    "map3_2": [[0, -np.pi]] * 6,
    "map3_4": [[0, -np.pi]] * 8 + [[1, 0]] * 7 + [[0.8, np.pi]] * 8,
    "map4_0": [[0.8, 0.75 * np.pi]] * 15,
    "map4_2": [[-1, 0.5 * np.pi]] * 18,
    "map4_4": [[0, -np.pi]] * 4,
    "map5,0": [[0, np.pi]] * 4,
    "map5_1": [[0, np.pi]] * 5,
    "map5_2": [[0, -np.pi]] * 4,
    "map5_4": [[0, np.pi]] * 7,
}

for map_name, task_info in task_dict.items():
    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    control_file_name = f"{map_name}_seed{seed}_start_{start_tile[0]},{start_tile[1]}_goal_{goal_tile[0]},{goal_tile[1]}.txt"
    f = open(dir_path + control_file_name)
    lines = f.readlines()

    tiles = []
    instructions = []

    for line in lines:
        ss = line.rstrip("\n").split(", ")
        tiles.append(eval(", ".join([ss[0], ss[1]])))
        instructions.append(ss[2])

    instructions = instructions[1:]
    print(map_name, seed, start_tile, goal_tile)
    print(tiles, instructions)

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

    _, _, done, _, info = env.step([0, 0])
    env.render()

    total_reward = 0
    total_step = 0
    actions = []

    assert tiles[0] == info['curr_pos']
    assert instructions[0] == "forward"

    if map_name in predefined_action_list:
        for action in predefined_action_list[map_name]:
            _, reward, _, truncated, info = env.step(action)
            # print(reward)
            total_reward += reward
            total_step += 1
            actions.append(action)
            env.render()

    idx = 0
    success = False
    while True:
        env_stack.clear()
        obs, _, _, _, info = env.step([0, 0])
        if info['curr_pos'] != tiles[idx]:
            break

        algo = algos[instructions[idx]]
        while info['curr_pos'] == tiles[idx]:
            action = algo.compute_single_action(
                observation=obs,
                explore=False,
            )
            obs, reward, done, truncated, info = env.step(action)
            # print(reward)
            total_reward += reward
            total_step += 1
            actions.append(action)
            env.render()

        idx += 1
        if idx == len(instructions):
            if info['curr_pos'] == tiles[idx]:
                success = True
            break

    if success:
        print("success")
        print(total_reward, total_step, total_reward / total_step)
        np.savetxt(f'./control_files/{map_name}_seed{seed}_start_{start_tile[0]},{start_tile[1]}_goal_{goal_tile[0]},{goal_tile[1]}.txt',
                   actions, delimiter=',')
    else:
        print("fail", env_old.map_name, env_old.cur_pos, tiles[idx], instructions[idx])
