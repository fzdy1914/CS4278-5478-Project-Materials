import json
import math
import time

import numpy as np
import ray
import torch
from gymnasium.wrappers import EnvCompatibility

from cnn_model import RegressionResNet
from find_highest_peak import find_highest_peak
from intelligent_robots_project import LaneFollower
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from torchvision import models, transforms

from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from path_generating import generate_path
from gym_duckietown.envs.directed_bot_env import goal_obj_position

delta_to_direction = {
    (1, 0): 0,
    (0, -1): 1,
    (-1, 0): 2,
    (0, 1): 3,
}


def launch_and_wrap_env(ctx):
    env = DirectedBotEnv(direction=ctx["direction"], domain_rand=False, max_steps=1500, map_name="map1_0", randomize_maps_on_reset=True)

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env


ray.init()
register_env("MyDuckietown", launch_and_wrap_env)

config = PPOConfig().environment("MyDuckietown", env_config={"direction": 3}).framework("torch").rollouts(num_rollout_workers=0).resources(num_gpus=0)
algo_forward_normal = config.build()
algo_forward_normal.restore("./forward_normal_result/final_best")

right_action = [[-0.8, np.pi]] * 15 + [[1, 0]] * 12
left_action = [[-0.9, -np.pi]] * 25

config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 1
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )
algo_left = config.build()
algo_left.restore("./left_result/final_best")

config = (
        PPOConfig()
        .environment("MyDuckietown", env_config={
            "direction": 2
        })
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )
algo_right = config.build()
algo_right.restore("./right_result/final_best")

algos = {
    "forward": algo_forward_normal,
    "left": algo_left,
    "right": algo_right,
}

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
    if "map4_0" != map_name:
        continue

    actions = []
    total_reward = 0
    total_step = 0

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    intentions = {start_tile: "forward"}

    print(map_name, seed, start_tile, goal_tile)

    env_old = DuckietownEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=map_name,
        seed=seed,
        user_tile_start=start_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False,
        my_mode="none",
    )

    env = EnvCompatibility(env_old)
    env = ResizeWrapper(env)
    env_stack = StackWrapper(env)
    env = NormalizeWrapper(env_stack)

    map_img, goal, start_pos = env_old.get_task_info()

    # start first tile handling
    robot = LaneFollower(intentions, map_img, goal, visualize=False)

    action = [0, 0]
    obs, _, _, info = env_old.step(action)
    env_old.render()

    mode = "Follower"
    while info["curr_pos"] == start_tile:
        if mode == "Follower":
            action = robot(obs, info, action)
            if robot.state_estimator.can_stop:
                mode = "RL"
                obs, _, _, _, info = env.step([0, 0])
                continue
            obs, reward, done, info = env_old.step(action)
            env_old.render()
        else:
            action = algo_forward_normal.compute_single_action(
                observation=obs,
                explore=False,
            )
            obs, reward, done, truncated, info = env.step(action)
            env_old.render()

    delta = (info["curr_pos"][0] - start_tile[0], info["curr_pos"][1] - start_tile[1])
    direction = delta_to_direction[delta]
    print(start_tile, info["curr_pos"], delta, direction)

    # forward, backward, left, right
    instructions = generate_path(map_img, info["curr_pos"], goal, direction)
    print(instructions)

    idx = 0
    success = False
    while True:
        env_stack.clear()
        obs, _, _, _, info = env.step([0, 0])
        if info['curr_pos'] != instructions[idx][0]:
            break

        if instructions[idx][1] == "backward":
            if instructions[idx + 1][1] == "right":
                for action in right_action:
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    total_step += 1
                    actions.append(action)
                    env.render()
            elif instructions[idx + 1][1] == "left":
                for action in left_action:
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    total_step += 1
                    actions.append(action)
                    env.render()
            elif instructions[idx + 1][1] == "forward":
                cur = info['curr_pos']
                while cur == info['curr_pos']:
                    obs, reward, done, truncated, info = env.step([0.65, np.pi])
                    env.render()
            else:
                print("unsupported")

            instructions[idx + 1][1] = "forward"
            idx += 1
            continue

        algo = algos[instructions[idx][1]]
        while info['curr_pos'] == instructions[idx][0]:
            action = algo.compute_single_action(
                observation=obs,
                explore=False,
            )
            if info['curr_pos'] == (6, 7):
                obs, reward, done, truncated, info = env.step([0.8, -np.pi])
            else:
                obs, reward, done, truncated, info = env.step(action)
            # print(reward, info)
            total_reward += reward
            total_step += 1
            actions.append(action)
            env.render()
        idx += 1
        if idx == len(instructions):
            if info['curr_pos'] == goal_tile:
                success = True
            break

    if success:
        print("success to goal")
    else:
        print("failed")
        continue

    # turning right
    obs_list = []
    reward_list = []
    for i in range(120):
        obs, reward, _, _ = env_old.step([0, -0.3])
        env.render()
        obs_list.append(transform(obs))
        reward_list.append(reward)

    obs_tenser = torch.stack(obs_list)
    output = model_angle(obs_tenser)

    scores = output.squeeze().tolist()
    idx = find_highest_peak(scores)

    if scores[idx] > 0.9:
        a = (idx + 1) // 10
        b = (idx + 1) % 10
        for i in range(a):
            actions.append([0, -3])
        if b > 0:
            actions.append([0, -0.3 * b])
        total_step += idx + 1
        total_reward += sum(reward_list[:idx + 1])
        for i in range(120 - idx - 1):
            env_old.step([0, 0.3])
            env.render()
    else:
        # reset to previous
        for i in range(12):
            env_old.step([0, 3])
            env.render()

        # turning left
        obs_list = []
        for i in range(120):
            obs, reward, _, _ = env_old.step([0, 0.3])
            env.render()
            obs_list.append(transform(obs))
            reward_list.append(reward)

        obs_tenser = torch.stack(obs_list)
        output = model_angle(obs_tenser)

        scores = output.squeeze().tolist()
        idx = find_highest_peak(scores)

        a = (idx + 1) // 10
        b = (idx + 1) % 10
        for i in range(a):
            actions.append([0, 3])
        if b > 0:
            actions.append([0, 0.3 * b])
        total_step += idx + 1
        total_reward += sum(reward_list[:idx + 1])
        for i in range(120 - idx - 1):
            env_old.step([0, -0.3])
            env.render()

    while True:
        obs, reward, done, info = env_old.step([1, 0])
        env.render()
        total_step += 1
        total_reward += reward
        actions.append([1, 0])
        image = transform(obs).unsqueeze(dim=0)
        dist = model_distance(image)[0][0]

        if dist < 0.38:
            break

    location = goal_obj_position[env_old.map_name]
    dist = math.sqrt((location[0] - env_old.cur_pos[0]) ** 2 + (location[1] - env_old.cur_pos[2]) ** 2)

    print(total_reward, total_step, total_reward / total_step, dist)
    np.savetxt(f'./m2_control_files/{map_name}_seed{seed}_start_{start_tile[0]},{start_tile[1]}_goal_{goal_tile[0]},{goal_tile[1]}.txt',
               actions, delimiter=',')
    # else:
    #     print("fail", env_old.map_name, env_old.cur_pos, tiles[idx], instructions[idx])

