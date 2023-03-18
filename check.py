from gym_duckietown.envs import SinGuidedBotEnv
from gym_duckietown.wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.utils import check_env


def launch_and_wrap_env(ctx):
    env = SinGuidedBotEnv(
        domain_rand=False,
        max_steps=10000,
        map_name="map1_0",
        randomize_maps_on_reset=True
    )

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env

env = launch_and_wrap_env(None)

check_env(launch_and_wrap_env(None))