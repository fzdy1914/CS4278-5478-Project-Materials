"""Module that provides a class to visualize things on the 2D map."""
import typing as T
import logging

import numpy as np
import cv2
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MapVisualizer:
    def __init__(
        self,
        map_image,
        goal_tile: T.Tuple[int, int] = None,
        lawful_states: T.List[T.Tuple[int, int, int]] = None,
        window_name: str = "DuckieMap",
        tile_pixel: int = 100,
        arrow_length: int = 30,
        color: T.Dict[str, T.Tuple[int, int, int]] = None,
    ):
        self._img = map_image
        self._window_name = window_name
        self._arrow_step_length = arrow_length / 2
        self._tile_pixel = tile_pixel
        self._color = (
            color
            if color is not None
            else {
                "current_state": (0, 155, 0),
                "goal_tile": (255, 0, 0),
            }
        )
        if goal_tile is not None:
            self._img = self._render_tile(
                goal_tile, self._img, self._color["goal_tile"]
            )

    def render(
        self,
        current_state: T.List[T.Tuple[int, int, T.Optional[int]]] = None,
    ):
        image = np.array(self._img)
        if current_state is not None:
            image = self._render_global_state(
                current_state, image, self._color["current_state"]
            )
        cv2.imshow(self._window_name, image)
        cv2.waitKey(1)

    def _render_global_state(self, state, new_image, color):
        center_x, center_y = self._center_of_tile(state)
        if state[2] is None:
            drawn_image = cv2.circle(new_image, (center_x, center_y), 10, color, -1)
            return drawn_image

        ang = np.pi * state[2] / 2
        arrow_length = 30
        x_step = int(arrow_length * np.cos(ang))
        y_step = int(arrow_length * np.sin(ang))
        start = (center_x - x_step, center_y - y_step)
        end = (center_x + x_step, center_y + y_step)
        drawn_image = cv2.arrowedLine(new_image, start, end, color, thickness=5)
        return drawn_image

    def _render_tile(self, tile, image, color):
        x, y = self._center_of_tile(tile)
        step = int(self._tile_pixel / 2)
        return cv2.rectangle(
            image, (x - step, y - step), (x + step, y + step), color, 10
        )

    @staticmethod
    def _center_of_tile(tile_coord):
        center_x = int(tile_coord[0] * 100 + 50)
        center_y = int(tile_coord[1] * 100 + 50)
        return (center_x, center_y)


class LocalVisualizer:
    def __init__(self):
        fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"][::-1])
        self.fig = fig
        self.ax = ax
        # Will be visualised as blue (because CV is BGR)
        (self.plt_line_white,) = ax.plot([0], [0], "r")
        # Will be visualised as red (because CV is BGR)
        (self.plt_line_yellow,) = ax.plot([0], [0], "b")
        (self.plt_wp,) = ax.plot([0], [0], "k")

        (self.plt_junction_forward,) = ax.plot([0], [0], "g")
        (self.plt_junction_right,) = ax.plot([0], [0], "g")

        (self.plt_init_red,) = ax.plot([0], [0], "g", marker="x")
        (self.plt_init_yellow,) = ax.plot([0], [0], "g", marker="o")

        ax.plot([0, 0], [-1, 2], "--", color="0.8")
        ax.plot([-1, 1], [0, 0], "--", color="0.8")
        ax.set_aspect(1)
        ax.set_xticks(np.arange(-1, 1, 0.1), minor=True)
        ax.set_yticks(np.arange(-1, 2, 0.1), minor=True)
        ax.set_xticks(np.arange(-1, 1, 0.5), minor=False)
        ax.set_yticks(np.arange(-1, 2, 0.5), minor=False)
        ax.grid(which="minor", alpha=0.2, color="green", linestyle="--", linewidth=0.2)
        ax.grid(which="major", alpha=0.5, color="green", linestyle="--", linewidth=0.5)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 2))

    def render(self, perception_output=None, waypoints=None, junction_output=None):
        if perception_output is not None:
            self.render_perception_data(*perception_output)
        if waypoints is not None:
            self.render_waypoints(waypoints)
        if junction_output is not None:
            self.render_junction_output(*junction_output)
        # Draw to figure and imshow
        self.fig.canvas.draw()
        out_img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        out_img = out_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        cv2.imshow("Top-down view", out_img)
        cv2.waitKey(1)

    def render_junction_output(self, forward_line, right_line):
        self.plt_junction_forward.set_xdata([0])
        self.plt_junction_forward.set_ydata([0])
        self.plt_junction_right.set_xdata([0])
        self.plt_junction_right.set_ydata([0])

        if forward_line is not None:
            self.plt_junction_forward.set_ydata(forward_line[0, :])
            self.plt_junction_forward.set_xdata(forward_line[1, :])
        if right_line is not None:
            self.plt_junction_right.set_ydata(right_line[0, :])
            self.plt_junction_right.set_xdata(right_line[1, :])

    def render_perception_data(
        self, white_line, yellow_line, junction_features, stopline_features
    ):
        self.plt_line_white.set_xdata([0])
        self.plt_line_white.set_ydata([0])
        self.plt_line_yellow.set_xdata([0])
        self.plt_line_yellow.set_ydata([0])
        self.plt_init_red.set_xdata([0])
        self.plt_init_red.set_ydata([0])
        self.plt_init_yellow.set_xdata([0])
        self.plt_init_yellow.set_ydata([0])

        # Visualise if we have something
        if white_line is not None:
            # Swap x y for visualization purpose
            self.plt_line_white.set_ydata(white_line[0, :])
            self.plt_line_white.set_xdata(white_line[1, :])
        if yellow_line is not None:
            self.plt_line_yellow.set_ydata(yellow_line[0, :])
            self.plt_line_yellow.set_xdata(yellow_line[1, :])
        if stopline_features is not None:
            centroid, _ = stopline_features
            self.plt_init_red.set_ydata([centroid[0, 0]])
            self.plt_init_red.set_xdata([centroid[1, 0]])
            # self.plt_init_yellow.set_ydata([initialising_features[0, 1]])
            # self.plt_init_yellow.set_xdata([initialising_features[1, 1]])

    def render_waypoints(self, waypoints):
        self.plt_wp.set_xdata([0])
        self.plt_wp.set_ydata([0])
        if len(waypoints) > 0:
            self.plt_wp.set_ydata(waypoints[0, :])
            self.plt_wp.set_xdata(waypoints[1, :])
