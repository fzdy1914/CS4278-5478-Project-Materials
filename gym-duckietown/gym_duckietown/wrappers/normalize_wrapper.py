import gymnasium
import numpy as np
from gymnasium import spaces


class NormalizeWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space[0].low[0, 0, 0]
        self.obs_hi = self.observation_space[0].high[0, 0, 0]
        obs_shape = self.observation_space[0].shape
        self.observation_space = spaces.Tuple([
            spaces.Box(0.0, 1.0, obs_shape, dtype=np.float64),
            spaces.Discrete(3),
        ])

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs[0] - self.obs_lo) / (self.obs_hi - self.obs_lo), obs[1]
