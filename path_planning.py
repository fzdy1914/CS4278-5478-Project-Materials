import math


class AStarPlanner:
    def __init__(self, obstacle_map):
        self.obstacle_map = obstacle_map
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = obstacle_map.shape
        self.x_width = self.max_x - self.min_x + 1

    class Node:
        def __init__(self, x, y, direction, control, cost, parent_index):
            self.x = x
            self.y = y

            self.direction = direction
            self.control = control
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return f"{self.x}, {self.y}, {self.direction}, {self.control}, {self.cost}, {self.parent_index}"

    def planning(self, sx, sy, sd, gx, gy):
        start_node = self.Node(sx, sy, sd, -1, 0.0, -1)
        goal_node = self.Node(gx, gy, -1, -1, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Empty Open Set")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                goal_node.direction = current.direction
                goal_node.control = current.control
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # dx, dy, direction, control, cost
            motions = self.get_motion_model(current.direction)

            # expand_grid search grid based on motion model
            for i, _ in enumerate(motions):
                node = self.Node(
                    x=current.x + motions[i][0],
                    y=current.y + motions[i][1],
                    direction=motions[i][2],
                    control=motions[i][3],
                    cost=current.cost + motions[i][4],
                    parent_index=c_id,
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        final_path = self.calc_final_path(goal_node, closed_set)

        return final_path

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        final_path = [(goal_node.y, goal_node.x)]
        direction_path = [goal_node.direction]
        control_path = [goal_node.control]

        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]

            final_path.append((n.y, n.x))
            direction_path.append(n.direction)
            control_path.append(n.control)

            parent_index = n.parent_index

        return final_path[::-1][:-1], control_path[::-1][1:]

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        if node.x < self.min_x:
            return False
        elif node.y < self.min_y:
            return False
        elif node.x >= self.max_x:
            return False
        elif node.y >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    @staticmethod
    def get_motion_model(direction):
        # dx, dy, direction
        init_motions = [
            [0, 1, 0],
            [-1, 0, 1],
            [0, -1, 2],
            [1, 0, 3],
        ]

        # direction: 0: east; 1: north; 2: west; 3: south
        motions = []

        for motion in init_motions:
            delta = motion[2] - direction
            if delta == 0:
                motions.append(motion + ["forward", 1])
            elif (delta == 1) or (delta == -3):
                motions.append(motion + ["left", 1.1])
            elif (delta == -1) or (delta == 3):
                motions.append(motion + ["right", 1.2])
            else:
                motions.append(motion + ["backward", 5])

        # dx, dy, direction, control, cost
        return motions
