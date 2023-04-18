import json
import math
import numpy as np

from gym_duckietown.envs import *
from gym_duckietown.envs.directed_bot_env import floor, new_ceil, ceil, new_floor


f = open("./testcases/milestone2.json", "r")
task_dict = json.load(f)


for map_name, task_info in task_dict.items():
    seed = task_info["seed"][0]

    start_tile = tuple(task_info["start"])
    goal_tile = tuple(task_info["goal"])

    print(map_name, seed, start_tile, goal_tile)

    env = DirectedBotEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=map_name,
        seed=seed,
        user_tile_start=goal_tile,
        goal_tile=goal_tile,
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

        goal_pos = env.goal_obj_position[env.map_name]
        if goal_pos[0] > env.cur_pos[0]:
            slope = (goal_pos[1] - env.cur_pos[2]) / (goal_pos[0] - env.cur_pos[0])
            aa = -np.arctan(slope)
        else:
            slope = (goal_pos[1] - env.cur_pos[2]) / (goal_pos[0] - env.cur_pos[0])
            aa = -np.arctan(slope) + np.pi
        aa = aa % (2 * np.pi)

        for i in range(360):
            r = 2 * np.pi / 360 * i
            env.cur_angle = r

            result_angle = min(math.fabs(aa - env.cur_angle), math.fabs(aa + 2 * np.pi - env.cur_angle))

            if result_angle > 0.5 * np.pi:
                continue

            if result_angle < 0.5:
                score = np.cos(result_angle * 3)
            else:
                score = 0

            obs = env.render_obs()
            # cv2.imwrite(f"dataset/{env.map_name}_{idx}_{i}_{score}.png", obs)
            with open(f"angle_dataset/{env.map_name}_{idx}_{i}_{score}.npy", 'wb') as f:
                np.save(f, obs)
                f.close()
