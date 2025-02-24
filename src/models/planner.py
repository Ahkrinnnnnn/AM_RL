import heapq
import numpy as np

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


class APlanner:
    def __init__(self, start, goal, obstacles, grid_size):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.open_list = []
        self.closed_list = set()
        self.grid = 0.5

    def heuristic(self, node):
        return max(abs(node.position[0] - self.goal.position[0]),
                   abs(node.position[1] - self.goal.position[1]),
                   abs(node.position[2] - self.goal.position[2]))
    
    def is_valid(self, position):
        for obstacle in self.obstacles:
            if not (self.grid_size[0][0] <= position[0] <= self.grid_size[0][1] and 
                    self.grid_size[1][0] <= position[1] <= self.grid_size[1][1] and
                    self.grid_size[2][0] < position[2] <= self.grid_size[2][1]):
                return False
            v = obstacle.get_verts().detach().numpy()
            min_pos = np.min(v, axis=0)
            max_pos = np.max(v, axis=0)
            if (min_pos[0] <= position[0] <= max_pos[0] and
                min_pos[1] <= position[1] <= max_pos[1] and
                min_pos[2] <= position[2] <= max_pos[2]):
                return False
        return True

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy, dz in self.grid * np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]):
            new_position = (node.position[0] + dx, node.position[1] + dy, node.position[2] + dz)
            if self.is_valid(new_position):
                neighbors.append(Node(new_position, parent=node))
        return neighbors

    def plan(self):
        self.start.g = 0
        self.start.h = self.heuristic(self.start)
        self.start.f = self.start.g + self.start.h
        heapq.heappush(self.open_list, self.start)

        while self.open_list:
            current_node = heapq.heappop(self.open_list)
            self.closed_list.add(current_node.position)

            if np.linalg.norm(np.array(current_node.position) - np.array(self.goal.position)) < self.grid:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return self.smoothen(path=path[::-1], num_points=50)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in self.closed_list:
                    continue

                neighbor.g = current_node.g + 0.1
                neighbor.h = self.heuristic(neighbor)
                neighbor.f = neighbor.g + neighbor.h

                if not any(node.position == neighbor.position for node in self.open_list):
                    heapq.heappush(self.open_list, neighbor)
                else:
                    for node in self.open_list:
                        if node.position == neighbor.position and node.g > neighbor.g:
                            node.g = neighbor.g
                            node.parent = neighbor.parent
                            node.f = neighbor.f

        return None
    
    def cubic_bezier_curve(self, p0, p1, p2, p3, t_values):
        t = t_values[:, np.newaxis]
        one_minus_t = 1 - t
        return (one_minus_t ** 3) * p0 + 3 * (one_minus_t ** 2) * t * p1 + 3 * (one_minus_t) * (t ** 2) * p2 + (t ** 3) * p3

    def smoothen(self, path, num_points):
        smoothed_path = []
        t_values = np.linspace(0, 1, num_points)
        for i in range(0, len(path)-3, 3):
            p0, p1, p2, p3 = path[i:i+4]
            bezier_points = self.cubic_bezier_curve(np.array(p0), np.array(p1), np.array(p2), np.array(p3), t_values)
            smoothed_path.extend(bezier_points)

        return np.array(smoothed_path)
    

    def vis(self, scene, path):
        scene.draw_debug_points(poss=path, colors=(0, 255, 0, 1))
