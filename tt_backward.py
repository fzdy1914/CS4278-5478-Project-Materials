import math
import time

import cv2
import gymnasium as gym
import numpy as np
from gym_duckietown.envs import DirectedBotEnv
from gym_duckietown.envs.directed_bot_env import floor, new_ceil, ceil, new_floor
from gymnasium.spaces import Box
from intelligent_robots_project import Perception, Initializer
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from scipy.signal import find_peaks
from torchvision import models, transforms
import torch
from cnn_model import RegressionResNet

percept = Perception()
initializer = Initializer()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = RegressionResNet(models.resnet50(pretrained=True), 1)
model.eval()
model.load_state_dict(torch.load("angle_model.pth"))

actions = []

env = DirectedBotEnv(
    domain_rand=False,
    max_steps=1500,
    map_name="map3_1",
    seed=4,
    user_tile_start="6,11",
    goal_tile="6,11",
    randomize_maps_on_reset=False,
    # my_mode="none",
    direction=5,
)

for idx, angle in enumerate([0, 0.5 * np.pi, np.pi, 1.5 * np.pi]):
    env.reset()

    if angle > 7 / 4 * np.pi or angle <= 1 / 4 * np.pi:
        op_x = floor
        op_y = new_ceil
    elif 1 / 4 * np.pi < angle <= 3 / 4 * np.pi:
        op_x = new_ceil
        op_y = ceil
    elif 3 / 4 * np.pi < angle <= 5 / 4 * np.pi:
        op_x = ceil
        op_y = new_floor
    else:
        op_x = new_floor
        op_y = floor
    env.cur_pos[0] = op_x(env.cur_pos[0])
    env.cur_pos[2] = op_y(env.cur_pos[2])

    env.cur_angle = angle

    if not env.valid_pose(env.cur_pos, env.cur_angle):
        continue

    obs, reward, done, info = env.step([0, 0])
    c = info["curr_pos"]
    total_reward = 0
    total_step = 0
    for i in range(15):
        env.step([-0.8, np.pi])
        env.render()
        print(env.get_grid_coords(env.cur_pos), total_step, reward)

    for i in range(12):
        env.step([1, 0])
        env.render()

    time.sleep(10)
    break