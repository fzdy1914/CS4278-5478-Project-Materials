import json
import math

import cv2
import numpy as np

from find_highest_peak import find_highest_peak
from gym_duckietown.envs import DirectedBotEnv
from gym_duckietown.envs.directed_bot_env import floor, new_ceil, ceil, new_floor
from intelligent_robots_project import Perception, Initializer
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
model_angle = RegressionResNet(models.resnet50(pretrained=True), 1)
model_angle.eval()
model_angle.load_state_dict(torch.load("angle_model.pth"))

model_distance = RegressionResNet(models.resnet50(pretrained=True), 1)
model_distance.eval()
model_distance.load_state_dict(torch.load("distance_model.pth"))


f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)


for map_name, task_info in task_dict.items():
    actions = []
    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    tiles = [start_tile]
    intentions = {start_tile: "forward"}

    print(map_name, seed, start_tile, goal_tile)

    env_old = DirectedBotEnv(
        domain_rand=False,
        max_steps=15000,
        map_name=map_name,
        seed=seed,
        user_tile_start=goal_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False,
        # my_mode="none",
        direction=5,
    )

    for idx, angle in enumerate([0, 0.5 * np.pi, np.pi, 1.5 * np.pi]):
        env_old.reset()

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
        env_old.cur_pos[0] = op_x(env_old.cur_pos[0])
        env_old.cur_pos[2] = op_y(env_old.cur_pos[2])

        env_old.cur_angle = angle

        if not env_old.valid_pose(env_old.cur_pos, env_old.cur_angle):
            continue

        # turning right
        obs_list = []
        for i in range(120):
            obs, _, _, _ = env_old.step([0, -0.3])
            obs_list.append(transform(obs))

        input = torch.stack(obs_list)
        output = model_angle(input)

        scores = output.squeeze().tolist()
        idx = find_highest_peak(scores)

        if scores[idx] > 0.9:
            a = (idx + 1) // 10
            b = (idx + 1) % 10
            for i in range(a):
                actions.append([0, -3])
            if b > 0:
                actions.append([0, -0.3 * b])
            for i in range(120 - idx - 1):
                env_old.step([0, 0.3])
        else:
            # reset to previous
            for i in range(12):
                env_old.step([0, 3])

            # turning left
            obs_list = []
            for i in range(120):
                obs, _, _, _ = env_old.step([0, 0.3])
                obs_list.append(transform(obs))

            input = torch.stack(obs_list)
            output = model_angle(input)

            scores = output.squeeze().tolist()
            idx = find_highest_peak(scores)

            a = (idx + 1) // 10
            b = (idx + 1) % 10
            for i in range(a):
                actions.append([0, 3])
            if b > 0:
                actions.append([0, 0.3 * b])
            for i in range(120 - idx - 1):
                env_old.step([0, -0.3])

        while True:
            obs, reward, done, info = env_old.step([1, 0])
            actions.append([1, 0])
            image = transform(obs).unsqueeze(dim=0)
            dist = model_distance(image)[0][0]

            if dist < 0.37:
                break

        print("is_done", done, reward, info)
        print(actions)
        location = env_old.goal_obj_position[env_old.map_name]
        dist = math.sqrt((location[0] - env_old.cur_pos[0]) ** 2 + (location[1] - env_old.cur_pos[2]) ** 2)
        print(dist)
        cv2.imwrite(env_old.map_name + "-" + str(idx) + ".png", env_old.render_obs())
