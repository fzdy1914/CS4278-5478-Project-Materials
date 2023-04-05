# Basic Lane Follower for CS4278/5478 Project

## Installation

Make sure you have `swig` with version `3.*.*` and a C-compiler installed on
your system. If you install `swig` by `conda`, run `conda install swig=3.*.*`;
if you use `homebrew` on Mac and conda cannot install `swig`, run `brew install
swig@3.0.12`.

You can clone this repo. In the repo, run:
``` sh
pip install .
```
Alternatively, you can directly install from this github repo:
``` sh
pip install git+https://github.com/AdaCompNUS/CS4278-5478-Project-Basic-Lane-Follower.git
```

To verify your installation, run
``` sh
intelligent-robots-project-example
```
You should see the robot moving and following the lane.

## Getting started

The above example is implemented [here](src/intelligent_robots_project/example.py).

Usage:
``` python
from intelligent_robots_project import LaneFollower
from gym_duckietown.envs import DuckietownEnv

intentions = {
    (1, 1): "forward",
    (1, 2): "forward",
    (1, 3): "forward",
    (1, 4): "forward",
    (1, 5): "forward",
    (1, 6): "forward",
    (1, 7): "left",
    (2, 7): "forward",
    (3, 7): "forward",
    (4, 7): "forward",
    (5, 7): "forward",
    (6, 7): "forward",
    (7, 7): "forward",
}

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1800,
    map_name="map2_1",
    seed=12,
    user_tile_start=(1, 1),
    goal_tile=(7, 7),
    randomize_maps_on_reset=False,
)
map_img, goal, start_pos = env.get_task_info()
robot = LaneFollower(intentions, map_img, goal, visualize=True)

action = [0, 0]
obs, reward, done, info = env.step(action)
action = robot(obs, info, action)
```
