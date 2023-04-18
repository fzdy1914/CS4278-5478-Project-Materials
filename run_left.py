import argparse
import time

from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from gymnasium.wrappers import EnvCompatibility

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from gym_duckietown.envs import *


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

ray.init()
register_env('MyDuckietown', launch_and_wrap_env)
config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 0
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training()
        .resources(num_gpus=0)
    )
algo = config.build()
algo.restore("D:\\left_result\\checkpoint_000646")

env = launch_and_wrap_env(None)

while True:
    obs, _, done, _, _ = env.step([0, 0])
    env.render()

    while not done:
        # Compute an action (`a`).
        action = algo.compute_single_action(
            observation=obs,
            explore=False,
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(action)
        print(action, reward)
        env.render()
        # Is the episode `done`? -> Reset.
    env.reset()
# time.sleep(100)
# algo.stop()