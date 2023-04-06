import math

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
from train_cnn import RegressionResNet

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
model.load_state_dict(torch.load("goal_detection.pth"))

actions = []

env = DirectedBotEnv(
    domain_rand=False,
    max_steps=1500,
    map_name="map5_3",
    seed=4,
    user_tile_start="12,15",
    goal_tile="12,15",
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

    # turning right
    obs_list = []
    for i in range(120):
        obs, _, _, _ = env.step([0, -0.3])
        obs_list.append(transform(obs))

    input = torch.stack(obs_list)
    output = model(input)

    scores = output.squeeze().tolist()
    peaks, _ = find_peaks(scores, distance = 5)
    highest_peak_index = peaks[np.argmax(np.array(scores)[peaks])]
    start = max(0, highest_peak_index - 1)
    end = min(len(scores), highest_peak_index + 2)
    peak_positions = np.arange(start, end)
    peak_scores = np.array(scores[start:end])
    mean = np.average(peak_positions, weights=peak_scores)
    idx = int(mean)

    if scores[idx] > 0.9:
        a = (idx + 1) // 10
        b = (idx + 1) % 10
        for i in range(a):
            actions.append([0, -3])
        if b > 0:
            actions.append([0, -0.3 * b])
        for i in range(120 - idx - 1):
            env.step([0, 0.3])
    else:
        # reset to previous
        for i in range(12):
            env.step([0, 3])

        # turning left
        obs_list = []
        for i in range(120):
            obs, _, _, _ = env.step([0, 0.3])
            obs_list.append(transform(obs))

        input = torch.stack(obs_list)
        output = model(input)

        scores = output.squeeze().tolist()
        peaks, _ = find_peaks(scores, distance=5)
        highest_peak_index = peaks[np.argmax(np.array(scores)[peaks])]
        start = max(0, highest_peak_index - 1)
        end = min(len(scores), highest_peak_index + 2)
        peak_positions = np.arange(start, end)
        peak_scores = np.array(scores[start:end])
        mean = np.average(peak_positions, weights=peak_scores)
        idx = int(mean)

        a = (idx + 1) // 10
        b = (idx + 1) % 10
        for i in range(a):
            actions.append([0, 3])
        if b > 0:
            actions.append([0, 0.3 * b])
        for i in range(120 - idx - 1):
            env.step([0, -0.3])

    while True:
        obs, _, done, _ = env.step([1, 0])
        env.render()
        actions.append([1, 0])

        perception_output = percept.step(obs, init=True)
        _, _, init_features, _ = perception_output
        if init_features is not None:
            _, hull = init_features
            if initializer.check_boundary_closeness(hull):
                break

    print(actions)
    location = env.goal_obj_position[env.map_name]
    dist = math.sqrt((location[0] - env.cur_pos[0]) ** 2 + (location[1] - env.cur_pos[2]) ** 2)
    print(dist)
    cv2.imwrite(env.map_name + ".png", env.render_obs())
    break