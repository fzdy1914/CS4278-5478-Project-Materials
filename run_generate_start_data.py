import json
import ray
from gymnasium.wrappers import EnvCompatibility
from intelligent_robots_project import LaneFollower
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper

import numpy as np
import pyglet
from gym_duckietown.envs import *
from pyglet.window import key
import sys
import cv2

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

result = {
    "map1_0": (0, 0),
    "map1_1": (0, 0),
    "map1_2": (0, 0),
    "map1_3": (0, 0),
    "map1_4": (np.pi, 0),
    "map2_0": (0, -1),
    "map2_1": (1.5 * np.pi, 0),
    "map2_2": (1.5 * np.pi, 0),
    "map2_3": (0.5 * np.pi, 0),
    "map2_4": (0.5 * np.pi, 0),
    "map3_0": (1.5 * np.pi, 0),
    "map3_1": (0, 0),
    "map3_2": (0, 0),
    "map3_3": (1.5 * np.pi, -1),
    "map3_4": (1.5 * np.pi, 0),
    "map4_0": (0, 0),
    "map4_1": (0, 0),
    "map4_2": (0, 0),
    "map4_3": (0.5 * np.pi, 0),
    "map4_4": (1.5 * np.pi, 0),
    "map5_0": (np.pi, 0),
    "map5_1": (0, 0),
    "map5_2": (0, 0),
    "map5_3": (0.5 * np.pi, 0),
    "map5_4": (1.5 * np.pi, -1),
}

m = "map5_4"

for map_name, task_info in task_dict.items():
    if m not in map_name:
        continue

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    if m in map_name:
        break


print(map_name, seed, start_tile, goal_tile)

env = DuckietownEnv(
    domain_rand=False,
    max_steps=150000,
    map_name=map_name,
    seed=seed,
    user_tile_start=start_tile,
    goal_tile=goal_tile,
    randomize_maps_on_reset=False,
    my_mode="none",
)
env.render()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost when pressing shift
    if key_handler[key.LSHIFT]:
        action *= 3

    obs, reward, done, info = env.step(action)
    # print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    print(env.cur_pos, env.cur_angle % (2 * np.pi), reward)

    env.render()
    if done:
        print(reward)
        env.reset()
        env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
