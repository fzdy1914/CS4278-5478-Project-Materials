import argparse
import time

from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from gymnasium.wrappers import EnvCompatibility

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from gym_duckietown.envs import *
from ray.rllib.algorithms.algorithm import Algorithm


def launch_and_wrap_env(ctx):
    env = DirectedBotEnv(
        direction=1,
        domain_rand=False,
        max_steps=100,
        map_name="map2_0",
        randomize_maps_on_reset=True
    )

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env


env = launch_and_wrap_env(None)
obs, _ = env.reset()
env.render()

register_env('MyDuckietown', launch_and_wrap_env)

algo = Algorithm.from_checkpoint("D:\\left_result\\checkpoint_000579")

num_episodes = 0
episode_reward = 0.0
done = False
while not done:
    # Compute an action (`a`).
    a = algo.compute_single_action(
        observation=obs,
        explore=False,
    )
    # Send the computed action `a` to the env.
    obs, reward, done, truncated, _ = env.step(a)
    episode_reward += reward
    env.render()
    # Is the episode `done`? -> Reset.

time.sleep(100)
algo.stop()