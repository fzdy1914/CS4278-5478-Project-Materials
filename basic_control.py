import numpy as np
import cv2
import math


def region_of_interest(image, vertices):
    # get region of interest
    mask = np.zeros_like(image)
    match_mask_color = 255

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines, color=(255, 0, 0), thickness=3):
    # draw lines on the image
    if lines is None:
        return
    image = np.copy(image)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    return image


def detect_lanes(image):
    height, width, _ = image.shape

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    region_of_interest_vertices = [(0, 50), (0, height), (width, height), (width, 50)]
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25
    )
    left_line_x, left_line_y = [], []
    right_line_x, right_line_y = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            # filter horizontal lines
            if math.fabs(slope) < 0.3:
                continue
            if slope < 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    lane_left, lane_right = [], []
    min_y, max_y = height * (1 / 5), height

    if left_line_x:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        lane_left = list(map(int, [poly_left(max_y), max_y, poly_left(min_y), min_y]))

        if lane_left[0] < 0:
            x1, y1, x2, y2 = lane_left
            lane_left[0] = 0
            lane_left[1] = int(-x1 * (y2 - y1) / (x2 - x1) + y1)

    if right_line_x:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        lane_right = list(map(int, [poly_right(max_y), max_y, poly_right(min_y), min_y]))

        if lane_right[0] > width:
            x1, y1, x2, y2 = lane_right
            lane_right[0] = width
            lane_right[1] = int((width - x1) * (y2 - y1) / (x2 - x1) + y1)

    return lane_left, lane_right


def calc_steering(lane_left, lane_right):

    height, width = 480, 640
    # detect two lanes
    if lane_left and lane_right:
        x1, x2 = lane_left[2], lane_right[2]
        mid = width / 2
        x_offset = (x1 + x2) / 2 - mid
    elif lane_left or lane_right:
        # x1, _, x2, _ = lane_left if lane_left else lane_right
        # x_offset = x2 - x1
        return 0.02
    else:
        return 0.02

    y_offset = height / 2
    steering = -math.atan(x_offset / y_offset)

    return steering


def trial_run(env):
    speed = 2e-01
    steering = 0
    start_pos = env.get_task_info()[2]

    while True:
        obs, reward, done, info = env.step([speed, steering])
        curr_pos = info["curr_pos"]
        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
        env.render()

        if start_pos != curr_pos:
            return (start_pos, curr_pos)

        lane_left, lane_right = detect_lanes(obs)
        steering = calc_steering(lane_left, lane_right)
