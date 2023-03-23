import argparse

import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
import sys
import cv2
import math
from basic_control import trial_run


def str2bool(v):
    """
    Reads boolean value from a string
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument("--max_steps", type=int, default=1500, help="max_steps")

# You should set them to different map name and seed accordingly
parser.add_argument("--map-name", "-m", default="map1_1", type=str)
parser.add_argument("--seed", "-s", default=0, type=int)
parser.add_argument("--start-tile", "-st", default="0,1", type=str, help="two numbers separated by a comma")
parser.add_argument("--goal-tile", "-gt", default="75,1", type=str, help="two numbers separated by a comma")
parser.add_argument(
    "--control_path", default="./map4_0_seed2_start_1,13_goal_3,3.txt", type=str, help="the control file to run"
)
parser.add_argument("--manual", default=False, type=str2bool, help="whether to manually control the robot")
parser.add_argument("--control", default=True, type=str2bool, help="whether to control the robot")

args = parser.parse_args()

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False,
)

env.render()

map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)

cv2.imshow("map", map_img)
cv2.waitKey(200)

# trail run part: using baisc control
if args.control:
    start_pos, curr_pos = trial_run(env)
    print(start_pos, curr_pos)

if args.manual:
    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    def update(dt):
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost when pressing shift
        if key_handler[key.LSHIFT]:
            action *= 3

        obs, reward, done, info = env.step(action)
        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()

else:
    # load control file
    actions = np.loadtxt(args.control_path, delimiter=",")

    for speed, steering in actions:
        obs, reward, done, info = env.step([speed, steering])
        curr_pos = info["curr_pos"]

        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")

        env.render()

    # dump the controls using numpy
    np.savetxt(
        f"./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt",
        actions,
        delimiter=",",
    )

env.close()
