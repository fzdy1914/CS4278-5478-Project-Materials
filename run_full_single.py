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


dir_path = "./testcases/milestone1_paths/"

control_file_name = "map1_2_seed2_start_2,1_goal_25,1.txt"
predefined_action_list = [[0, -np.pi]] * 5

blocks = control_file_name.rstrip(".txt").split("_")

map_name = blocks[0] + "_" + blocks[1]

seed = int(blocks[2].split("seed")[1])

start_tile = tuple((int(i) for i in blocks[4].split(",")))
goal_tile = tuple((int(i) for i in blocks[6].split(",")))

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
algo_forward_normal.restore("./forward_normal_result/final_hard_best")

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
algo_left.restore("./left_result/final_best")

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

obs, _, done, _, info = env.step([0, 0])
env.render()

time.sleep(2)

total_reward = 0
total_step = 0
actions = []

assert tiles[0] == info['curr_pos']
assert instructions[0] == "forward"

# this looks like a hack, but we believe solving it belongs the scope of milestone 2
# we will rotate and find one available tile to go in order to know the angle of the robot.
# The route planning will then based on the new angle.
# However, in milestone 1, the route is fixed, thus, we have to do this to avoid the issue of bad spawning.
for action in predefined_action_list:
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
        # print(reward, env_old.cur_pos)
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
    print("fail")
