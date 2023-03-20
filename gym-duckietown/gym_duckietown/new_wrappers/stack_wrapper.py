import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StackWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=4):
        super(StackWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)

        # The last dimension, is used. For images, this should be the depth.
        # For vectors, the output is still a vector, just concatenated.
        self.buffer_axis = len(obs_space_shape_list) - 1
        obs_space_shape_list[self.buffer_axis] *= obs_buffer_depth

        limit_low = self.observation_space.low[0, 0, 0]
        limit_high = self.observation_space.high[0, 0, 0]

        self.observation_space = spaces.Box(
                limit_low,
                limit_high,
                tuple(obs_space_shape_list),
                dtype=self.observation_space.dtype)

        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        if self.obs_buffer_depth == 1:
            return obs
        if self.obs_buffer is None:
            self.obs_buffer = np.concatenate(tuple([obs for _ in range(self.obs_buffer_depth)]), axis=self.buffer_axis)
        else:
            self.obs_buffer = np.concatenate((self.obs_buffer[..., (obs.shape[self.buffer_axis]):], obs), axis=self.buffer_axis)
        return self.obs_buffer

    def reset(self, **kwargs):
        self.obs_buffer = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
