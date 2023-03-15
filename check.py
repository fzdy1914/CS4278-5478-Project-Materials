from gym_duckietown.envs import GuidedBotEnv
from gym_duckietown.wrappers import NormalizeWrapper, ResizeWrapper
from gymnasium.spaces import Discrete
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.utils import check_env


def launch_and_wrap_env(ctx):
    env = GuidedBotEnv(
        domain_rand=False,
        max_steps=10000,
        map_name="map1_0",
        randomize_maps_on_reset=True
    )

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)

    return env

env = launch_and_wrap_env(None)
result = env.reset()
obs, info = result

print(Discrete(2).contains(1))

check_env(launch_and_wrap_env(None))