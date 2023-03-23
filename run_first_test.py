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

control_file_name = "map5_4_seed1_start_3,3_goal_3,10.txt"
predefined_action_list = [[0, np.pi]] * 8

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
    time.sleep(0.05)
