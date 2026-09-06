import numpy as np
import math
import cvxpy as cp
from simulator import centerline

def sign(x):
    return 1 if x >= 0 else -1

def dist_euclid(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def pt_dist(pt1, pt2):
    return dist_euclid(pt1[0], pt1[1], pt2[0], pt2[1])

class WaypointManager:
    def __init__(self, waypoint_count:int=100, track_len=105):
        self.ds = track_len / waypoint_count
        self.waypoint_count = waypoint_count
        self.centerline_discrete = np.array([centerline(s) for s in np.linspace(0, track_len, waypoint_count)])
        self.last_centerline_idx = 0
        self.normals = None

    def search_for_centerline_goalpoint(self, current_x, current_y, lookahead_dist):
        # referenced from
        # https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
        if self.last_centerline_idx > len(self.centerline_discrete):
            self.last_centerline_idx = 0
        start_idx = self.last_centerline_idx

        path = self.centerline_discrete
        goal_pt = path[self.last_centerline_idx]

        # line-circle intersection, checking each line segment along waypoint list, one at a time
        for i in range(start_idx, self.waypoint_count + 10):
            idx = i
            if idx >= len(path) - 1:
                idx = i % (len(path) - 1)
            # assumes circle is centered on 0, 0, so correct for offset
            x1 = path[idx][0] - current_x
            y1 = path[idx][1] - current_y
            x2 = path[idx + 1][0] - current_x
            y2 = path[idx + 1][1] - current_y
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx ** 2 + dy ** 2)
            D = x1 * y2 - x2 * y1
            discriminant = (lookahead_dist ** 2) * (dr ** 2) - D ** 2
            if discriminant >= 0:
                sol_x1 = (D * dy + sign(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_x2 = (D * dy - sign(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2

                # true solution points must be re-offset
                sol_pt1 = [sol_x1 + current_x, sol_y1 + current_y]
                sol_pt2 = [sol_x2 + current_x, sol_y2 + current_y]

                # discriminant check only considers infinite line, must restrict to segment
                min_x = min(path[idx][0], path[idx + 1][0])
                min_y = min(path[idx][1], path[idx + 1][1])
                max_x = max(path[idx][0], path[idx + 1][0])
                max_y = max(path[idx][1], path[idx + 1][1])
                # if one or both of the solutions are in range
                if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) or (
                        (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):

                    # if both solutions are in range, check which one is better
                    if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) and (
                            (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):
                        # find the point further down
                        if pt_dist(sol_pt1, path[idx + 1]) < pt_dist(sol_pt2, path[idx + 1]):
                            goal_pt = sol_pt1
                        else:
                            goal_pt = sol_pt2
                    # if not both solutions are in range, take the one that's in range
                    else:
                        if (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y):
                            goal_pt = sol_pt1
                        else:
                            goal_pt = sol_pt2
                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if pt_dist(goal_pt, path[idx + 1]) < pt_dist([current_x, current_y], path[idx + 1]):
                        # update self.last_centerline_idx and exit
                        # sanity check: don't loop around early
                        if (idx - self.last_centerline_idx) < len(self.centerline_discrete) / 2:
                            self.last_centerline_idx = idx
                            break
                    else:
                        # can't find intersection in next segment, but don't look in this one
                        self.last_centerline_idx = idx + 1
                # no solutions are in range
                else:
                    goal_pt = path[self.last_centerline_idx]
        return goal_pt, self.last_centerline_idx

    def generate_raceline(self, search_width: tuple[float, float]=(-0.2, 0.2)):
        normals_temp = []
        for i in range(self.waypoint_count):
            idx_next = (i + 1) % self.waypoint_count
            idx_prev = (i - 1) % self.waypoint_count
            pt_next = self.centerline_discrete[idx_next]
            pt_prev = self.centerline_discrete[idx_prev]

            dy = pt_next[1] - pt_prev[1]
            dx = pt_next[0] - pt_prev[0]

            # https://stackoverflow.com/questions/1243614/how-do-i-calculate-the-normal-vector-of-a-line-segment
            normal = np.array([-dy, dx])
            normal /= np.linalg.norm(normal)
            normals_temp.append(normal)

        self.normals = np.array(normals_temp)

        # displacement factor
        alpha = cp.Variable(self.waypoint_count)

        x_shifted = self.centerline_discrete[:, 0] + cp.multiply(alpha, self.normals[:, 0])
        y_shifted = self.centerline_discrete[:, 1] + cp.multiply(alpha, self.normals[:, 1])

        cost = 0
        # need to work within cvxpy operations
        for i in range(self.waypoint_count):
            idx_next = (i + 1) % self.waypoint_count
            idx_prev = (i - 1) % self.waypoint_count

            # cannot use kappa for minimum curvature QP optimization because breaks DCP
            # (cannot divide by optimization variable)
            d2x = x_shifted[idx_next] - 2 * x_shifted[i] + x_shifted[idx_prev]
            d2y = y_shifted[idx_next] - 2 * y_shifted[i] + y_shifted[idx_prev]
            cost = cost + cp.square(d2x) + cp.square(d2y)

        constraints = [alpha >= search_width[0], alpha <= search_width[1]]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        new_alpha = alpha.value
        print(alpha.value)
        optimal_x = self.centerline_discrete[:, 0] + new_alpha * self.normals[:, 0]
        optimal_y = self.centerline_discrete[:, 1] + new_alpha * self.normals[:, 1]
        self.centerline_discrete = np.vstack((optimal_x, optimal_y)).T