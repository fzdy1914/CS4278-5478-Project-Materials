import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env


class ContinuousEnv(gym.Env):
   def __init__(self):
       self.action_space = Box(
           low=np.array([0]),
           high=np.array([1]),
           dtype=np.float64
       )
       self.observation_space = Box(0.0, 1.0, shape=(1, ))

   def reset(self, **kwargs):
       return [0.0], {}

   def step(self, action):
       return [0.0], 1.0, False, False, {}


register_env('ContinuousEnv', lambda ctx: ContinuousEnv())

config = (
    PPOConfig()
    # or "corridor" if registered above
    .environment("ContinuousEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training()
)

algo = config.build()
algo.train()
