import numpy as np
from path_planning import AStarPlanner


def generate_path(map_img, start_pos, goal_pos, init_d):

    obstacle_map = parse_map(map_img)
    a_star = AStarPlanner(obstacle_map)
    sy, sx = start_pos
    gy, gx = goal_pos
    sd = init_d
    direction_path = a_star.planning(sx, sy, sd, gx, gy)

    # direction: 0: east; 1: north; 2: west; 3: south
    control_path = []

    for i in range(len(direction_path) - 1):
        delta = direction_path[i + 1] - direction_path[i]
        if delta == 0:
            control_path.append("forward")
        elif (delta == 1) or (delta == -3):
            control_path.append("left")
        elif (delta == -1) or (delta == 3):
            control_path.append("right")
        else:
            control_path.append("backward")

    return control_path


def parse_map(map_img):
    height, width, _ = map_img.shape
    num_row, num_col = map(lambda x: x // 100, [height, width])
    obstacle_map = []
    ratio = 0.8

    for i in range(num_row):
        map_row = []
        for j in range(num_col):
            tile_img = map_img[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100, :]
            non_zero_num = np.count_nonzero(tile_img)
            if non_zero_num <= 100 * 100 * ratio:
                map_row.append(True)
            else:
                map_row.append(False)
        obstacle_map.append(map_row)

    return np.array(obstacle_map)
