import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(84, 84)):
        super(ResizeWrapper, self).__init__(env)
        self.shape = shape + (self.observation_space[0].shape[2],)
        self.observation_space = spaces.Tuple([
            spaces.Box(
                self.observation_space[0].low[0, 0, 0],
                self.observation_space[0].high[0, 0, 0],
                self.shape,
                dtype=self.observation_space[0].dtype),
            spaces.Discrete(3),
        ])

    def observation(self, observation):
        resized = cv2.resize(observation[0], self.shape[:2][::-1], interpolation=cv2.INTER_AREA, )
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, 2)
        return resized, observation[1]
