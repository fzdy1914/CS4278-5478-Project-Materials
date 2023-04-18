import json

import numpy as np
import ray
import torch

from cnn_model import RegressionResNet
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from torchvision import models, transforms

from gymnasium.wrappers import EnvCompatibility
from gym_duckietown.envs import *
from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper


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
algo_forward_normal.restore("./forward_normal_result/final_hard_best")

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
algo_left.restore("./left_result/final_hard_best")
#
# config = (
#         PPOConfig()
#         .environment("MyDuckietown", env_config={
#             "direction": 2
#         })
#         .framework("torch")
#         .rollouts(num_rollout_workers=0)
#         .resources(num_gpus=0)
#     )
# algo_right = config.build()
# algo_right.restore("./right_result/final_hard_best")
#
# algos = {
#     "forward": algo_forward_normal,
#     "left": algo_left,
#     "right": algo_right,
# }

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model_start_angle = RegressionResNet(models.resnet50(pretrained=True), 1)
model_start_angle.eval()
model_start_angle.load_state_dict(torch.load("start_angle_model.pth"))

model_start_tile = RegressionResNet(models.resnet50(pretrained=True), 1)
model_start_tile.eval()
model_start_tile.load_state_dict(torch.load("start_tile_model.pth"))

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

for map_name, task_info in task_dict.items():
    # if "map1" in map_name:
    #     continue

    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    tiles = [start_tile]
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

    action = [0, 0]
    obs, _, _, info = env_old.step(action)
    env_old.render()

    image = transform(obs).unsqueeze(dim=0)
    score = model_start_angle(image)[0][0]
    obs_list = []
    obs_list.append(transform(obs))
    count = 0
    if score < 0.97:
        angle = "left"
        obs, _, _, info = env_old.step([0, 3])
        env_old.render()
        image = transform(obs).unsqueeze(dim=0)
        new_score = model_start_angle(image)[0][0]
        if new_score > score:
            count = 10
            score = new_score
            obs_list.append(transform(obs))
        else:
            obs, _, _, info = env_old.step([0, -0.3])
            env_old.render()
            angle = "right"

        if angle == "left":
            action = [0, 0.3]
        else:
            action = [0, -0.3]

        while score < 0.97:
            obs, _, _, info = env_old.step(action)
            obs_list.append(transform(obs))
            env_old.render()
            image = transform(obs).unsqueeze(dim=0)
            score = model_start_angle(image)[0][0]
            count += 1

    obs_tenser = torch.stack(obs_list[:30])
    output = model_start_tile(obs_tenser)
    tile = torch.mean(output)
    if tile < -0.5:
        algo = algo_left
    else:
        algo = algo_forward_normal

    obs, _, _, _, info = env.step([0, 0])
    while info["curr_pos"] == start_tile:
        action = algo.compute_single_action(
            observation=obs,
            explore=False,
        )
        obs, reward, done, truncated, info = env.step(action)
        env_old.render()
