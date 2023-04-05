"""This module provides high-level planning functionality."""
import typing as T
import logging

import numpy as np

logger = logging.getLogger(__name__)


def rotation_world_local(orientation: int):
    ang = orientation * np.pi / 2
    return np.array(
        [
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ]
    )


def intrd(val):
    return int(round(val))


class MapNode:
    def __init__(self, coord: T.Tuple[int, int, int], kind: str = "<unspecified>"):
        self.coord = coord
        self.kind = kind

    def __repr__(self):
        return f"MapNode(coord={self.coord}, kind={self.kind})"


def coord2index(x, y):
    """Convert coordinate to index."""
    return intrd(x) + 1, intrd(y) + 1


def index2coord(xi, yi):
    """Convert index to coordinate."""
    return xi - 1, yi - 1


def construct_nodes_from_3by3(
    x,
    y,
    map_patch_in_offsetworld: np.ndarray,
    x_upper_open_lim=1000,
    y_upper_open_lim=1000,
) -> T.List[MapNode]:
    """Construct node for a path given its 8 neighbors.

    Define a reference frame offset-world:
        - all three axes are aligned
        - origin offseted such that the robot position is (0, 0) in this frame

    Define a reference frame local:
        - x-axis is aligned with the robot pointing direction
        - z-axis is aligned with that of the world and offset-world frames
          (pointing to the ground)

    Args:
        x: global x coordinate.
        y: global y coordinate.
        map_patch_in_offsetworld: a 3x3 map expressed in the offset-world frame.
        x_upper_open_lim: Upper limit of x coord.
        y_upper_open_lim: Upper limit of y coord.
    """

    assert map_patch_in_offsetworld[coord2index(0, 0)] == 1  # must be a road
    nodes = []
    for orientation in range(4):
        # in local coordinate
        local = convert_3by3_map_to_local_axis(map_patch_in_offsetworld, orientation)
        node = MapNode((x, y, orientation))
        # Move forward
        if local[coord2index(1, 0)] == 1:
            node.kind = "straight"
        # Invalid
        if local[coord2index(-1, 0)] == 0:
            continue
        # Move right (note z-axis direction)
        if local[coord2index(0, 1)] == 1:
            node.kind = "rturn"
        # Move left
        if local[coord2index(0, -1)] == 1:
            node.kind = "lturn"
        # Check if is a junction, including T junctions
        if (
            local[coord2index(1, 0)]
            + local[coord2index(0, 1)]
            + local[coord2index(0, -1)]
            >= 2
        ):
            node.kind = "junction"
            # Can U turn
        nodes.append(node)
    return nodes


def convert_3by3_map_to_local_axis(
    map_patch_in_offsetworld: np.ndarray, orientation: int
):
    R_world_local = rotation_world_local(orientation)
    map_local = np.zeros([3, 3], dtype=int)
    for xi in range(3):
        for yi in range(3):
            if map_patch_in_offsetworld[xi, yi] == 0:
                continue
            v_world = np.array(index2coord(xi, yi))
            v_local = R_world_local.T @ v_world
            map_local[coord2index(v_local[0], v_local[1])] = 1
    return map_local


def construct_lawful_graph(
    map_image, tile_width: int = 100, threshold: float = 100
) -> T.Dict[T.Tuple[int, int, int], MapNode]:
    """Construct a graph from the 2D image of the map.

    The nodes in the graph are the global states that do not violate the traffic
    rule. See the definitions in the package README.md.
    """
    x_count = intrd(map_image.shape[0] / tile_width)
    y_count = intrd(map_image.shape[1] / tile_width)
    logger.info(
        "Received a %s map image, x %s by y %s", map_image.shape, x_count, y_count
    )
    grid = []  # 1 for road, 0 for obstacle
    for x in range(x_count):
        grid.append([])
        x_start = x * tile_width
        x_end = min((x + 1) * tile_width, map_image.shape[0])
        for y in range(y_count):
            y_start = y * tile_width
            y_end = min((y + 1) * tile_width, map_image.shape[1])
            ave = map_image[x_start:x_end, y_start:y_end, :].mean()
            if ave < threshold:
                grid[x].append(0)
            else:
                grid[x].append(1)
    grid = np.array(grid).T  # so that 1st dim is x, 2nd dim is y
    x_count = grid.shape[0]
    y_count = grid.shape[1]
    logger.info("Generate grid (1st dim x, 2nd dim y):\n%s", grid)

    padded_grid = np.ones([x_count + 2, y_count + 2])
    padded_grid[1 : x_count + 1, 1 : y_count + 1] = grid

    ret = {}
    for x in range(x_count):
        for y in range(y_count):
            if padded_grid[x + 1, y + 1] == 0:
                continue
            nodes = construct_nodes_from_3by3(
                x, y, padded_grid[x : x + 3, y : y + 3], x_count, y_count
            )
            for n in nodes:
                logger.debug("Constructed node %s", n)
                ret[n.coord] = n
    return ret
