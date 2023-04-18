import json
import math
import cv2
import numpy as np

from gym_duckietown.envs import *

f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)

result = {
    "map1_0": (0, 0),
    "map1_1": (0, 0),
    "map1_2": (0, 0),
    "map1_3": (0, 0),
    "map1_4": (np.pi, 0),
    "map2_0": (0, -1),
    "map2_1": (1.5 * np.pi, 0),
    "map2_2": (1.5 * np.pi, 0),
    "map2_3": (0.5 * np.pi, 0),
    "map2_4": (0.5 * np.pi, 0),
    "map3_0": (1.5 * np.pi, 0),
    "map3_1": (0, 0),
    "map3_2": (0, 0),
    "map3_3": (1.5 * np.pi, -1),
    "map3_4": (1.5 * np.pi, 0),
    "map4_0": (0, 0),
    "map4_1": (0, 0),
    "map4_2": (0, 0),
    "map4_3": (0.5 * np.pi, 0),
    "map4_4": (1.5 * np.pi, 0),
    "map5_0": (np.pi, 0),
    "map5_1": (0, 0),
    "map5_2": (0, 0),
    "map5_3": (0.5 * np.pi, 0),
    "map5_4": (1.5 * np.pi, -1),
}

for map_name, task_info in task_dict.items():
    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    print(map_name, seed, start_tile, goal_tile)

    env = DuckietownEnv(
        domain_rand=False,
        max_steps=150000,
        map_name=map_name,
        seed=seed,
        user_tile_start=start_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False,
        my_mode="none",
    )

    target = result[map_name]

    for i in range(360):
        r = 2 * np.pi / 360 * i
        env.cur_angle = r

        result_angle = min(math.fabs(target[0] - env.cur_angle), math.fabs(target[0] + 2 * np.pi - env.cur_angle))

        score = np.cos(result_angle)

        obs = env.render_obs()
        # cv2.imwrite(f"start_angle_dataset/{env.map_name}_{i}_{score}.png", obs)
        with open(f"start_angle_dataset/{env.map_name}_{i}_{score}.npy", 'wb') as f:
            np.save(f, obs)
            f.close()

        # cv2.imwrite(f"start_tile_dataset/{env.map_name}_{i}_{target[1]}.png", obs)
        with open(f"start_tile_dataset/{env.map_name}_{i}_{target[1]}.npy", 'wb') as f:
            np.save(f, obs)
            f.close()