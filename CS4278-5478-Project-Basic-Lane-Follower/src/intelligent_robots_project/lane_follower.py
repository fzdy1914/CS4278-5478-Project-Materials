"""Module that contains the robot class."""
import typing as T
import logging

from .state_estimator import StateEstimator
from .map import construct_lawful_graph
from .visualizer import LocalVisualizer, MapVisualizer
from .perception import Perception
from .controller import (
    SimpleController,
    TrajectoryGenerator,
    convert_tile_traj_to_bot_frame,
)


logger = logging.getLogger(__name__)


class Initializer:
    def __init__(self):
        self.rotation_dir = None
        self.too_close = False
        self.close_counter = 10

    def _get_rotation_dir(self, init_feat):
        return -1 if init_feat[0] * init_feat[1] < 0 else 1

    def _check_boundary_closeness(self, hull):
        for i in range(hull.shape[1]):
            j = i + 1 if i < hull.shape[1] - 1 else 0
            if abs(hull[1, i] - hull[1, j]) < 3 and abs(hull[0, i] - hull[0, j]) > 400:
                return True
        return False

    def get_action(self, perception_output):
        _, _, init_features, _ = perception_output

        # Check how close we are to the boundary. If we are too close, force a reversal
        if self.too_close and self.close_counter > 0:
            self.close_counter -= 1
            return [-1.0, 0]
        if self.too_close and self.close_counter == 0:
            self.close_counter = 5
            self.too_close = False
        if not self.too_close and init_features is not None:
            _, hull = init_features
            self.too_close = self._check_boundary_closeness(hull)
            logger.info("Closeness check: %s", self.too_close)

        # If we are not too close, we determine which way to pivot judging by the slant of the lane
        if self.rotation_dir is None:
            if init_features is not None:
                lane_dir, _ = init_features
                self.rotation_dir = self._get_rotation_dir(lane_dir)
                logger.info("Initialised rotation_dir: %s", self.rotation_dir)
                sign = self.rotation_dir
            else:
                sign = 1
        else:
            sign = self.rotation_dir

        logger.info("Rotation direction: %s", sign)
        action = [0, sign * -1.5]
        return action


class LaneFollower:
    """The class that encapsulates the task logic."""

    def __init__(
        self,
        intentions,
        map_img,
        goal,
        visualize: bool = True,
    ):
        # classes
        self.state_estimator = StateEstimator(dt=1 / 22.0)
        self.map_tiles = construct_lawful_graph(map_img)
        self.map_visualizer = MapVisualizer(map_img, goal_tile=goal)
        self.local_visualizer = LocalVisualizer()
        self.percept = Perception(visualise_topdown=False)
        self.controller = SimpleController()
        self.traj_gen = TrajectoryGenerator(y_ref=0.7)

        # variables
        self.command = None

        self.initializer = Initializer()

        self.goal = goal
        self.intentions = intentions
        self.should_visualize = visualize

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def act(self, observation, info: T.Dict[str, T.Any], prev_action) -> T.List[float]:
        """Compute the action from sensor data.

        Args:
            observation: The camera image. The first element of DuckietownEnv.step. 480 x 640 x 3.
            info: The additional data. The third element of DuckietownEnv.step
                info[curr_pos]: The tile index (i: int, j: int)
            prev_action: The previous action
        """
        self.state_estimator.update_action(prev_action)
        has_changed = self.state_estimator.update_gps(info["curr_pos"])
        if has_changed:
            if self.state_estimator.global_state[2] is not None:
                current_node_kind = self.map_tiles[
                    tuple(self.state_estimator.global_state)
                ].kind
                self.command = self.intentions[
                    tuple(self.state_estimator.global_state[0:2])
                ]
                self.state_estimator.update_plan(current_node_kind, self.command)
            else:
                self.state_estimator.update_plan(None, None)
                self.command = None
            if self.should_visualize:
                self.map_visualizer.render(
                    current_state=self.state_estimator.global_state,
                )

        is_init = self.state_estimator.local_state is None
        reached_goal_tile = (
            self.state_estimator.global_state[0] == self.goal[0]
            and self.state_estimator.global_state[1] == self.goal[1]
        )
        perception_output = self.percept.step(
            observation, init=is_init, is_final=reached_goal_tile
        )

        if reached_goal_tile:
            logger.info("Reached goal! Done.")
            return (0, 0)

        self.state_estimator.update_perception(perception_output)
        if self.should_visualize:
            self.local_visualizer.render(perception_output=perception_output)
        if self.state_estimator.local_state is None:
            action = self.initializer.get_action(perception_output)
            if self.should_visualize:
                self.local_visualizer.render(waypoints=[])
        else:
            state = self.state_estimator.local_state
            tile_waypoints = self.traj_gen(state, self.command)
            self.controller.set_trajectory(tile_waypoints)
            action = self.controller(state)
            body_waypoints = convert_tile_traj_to_bot_frame(state, tile_waypoints)
            if self.should_visualize:
                self.local_visualizer.render(waypoints=body_waypoints)

        return action
