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

def menger(pt1, pt2, pt3):
    s1 = pt_dist(pt1, pt2)
    s2 = pt_dist(pt2, pt3)
    s3 = pt_dist(pt1, pt3)
    sp = (s1 + s2 + s3) / 2
    a = math.sqrt(sp * (sp - s1) * (sp - s2) * (sp - s3))
    denom = (s1 * s2 * s3)
    # quick and dirty divby0 fix
    if denom == 0:
        denom = 0.00001
    kappa = (4 * a) / denom
    if kappa == 0:
        r = 1000000
    else:
        r = 1 / kappa
    return kappa, r

class WaypointManager:
    def __init__(self, waypoint_count:int=100, track_len=104):
        self.ds = track_len / waypoint_count
        self.waypoint_count = waypoint_count
        self.centerline_discrete = np.array([centerline(s) for s in np.linspace(0, track_len, waypoint_count)])
        self.last_idx = 0
        self.raceline = None
        self.normals = None

    def search_for_goalpoint(self, current_x: float, current_y: float, lookahead_dist: float, use_raceline=True):
        # referenced from
        # https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
        path = self.centerline_discrete
        if use_raceline and self.raceline is not None:
            path = self.raceline

        if self.last_idx > len(self.centerline_discrete):
            self.last_idx = 0
        start_idx = self.last_idx
        goal_pt = path[self.last_idx]

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
                # solve for intersection
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
                        # update self.last_idx and exit
                        # sanity check: don't loop around early
                        if (idx - self.last_idx) < len(self.centerline_discrete) / 2:
                            self.last_idx = idx
                            break
                    else:
                        # can't find intersection in next segment, but don't look in this one
                        self.last_idx = idx + 1
                # no solutions are in range
                else:
                    goal_pt = path[self.last_idx]
        return goal_pt, self.last_idx

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

            # not using kappa for minimum curvature QP optimization because breaks DCP
            # (cannot divide by optimization variable)
            # TODO: test using actual curvature with DNLP flag enabled, or change solver
            d2x = x_shifted[idx_next] - 2 * x_shifted[i] + x_shifted[idx_prev]
            d2y = y_shifted[idx_next] - 2 * y_shifted[i] + y_shifted[idx_prev]
            cost = cost + cp.square(d2x) + cp.square(d2y)

        constraints = [alpha >= search_width[0], alpha <= search_width[1]]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        new_alpha = alpha.value
        optimal_x = self.centerline_discrete[:, 0] + new_alpha * self.normals[:, 0]
        optimal_y = self.centerline_discrete[:, 1] + new_alpha * self.normals[:, 1]
        self.raceline = np.vstack((optimal_x, optimal_y)).T

    def get_optimal_states(self, current_x: float, current_y: float, lookahead_dist: float, window: int,
                           use_raceline :bool=True, max_a_centr: float=5, v_target: tuple[float, float]=(0,12)):
        starting_waypoint, starting_idx = self.search_for_goalpoint(current_x, current_y, lookahead_dist, use_raceline)
        path = self.raceline if (use_raceline and self.raceline is not None) else self.centerline_discrete
        idx_curr = starting_idx
        states = []
        for i in range(window):
            idx_next = (idx_curr + i) % len(path)
            idx_prev = starting_idx - 1
            if idx_prev < 0:
                idx_prev = len(path) - 1

            prev_pt = path[idx_prev]
            curr_pt = path[idx_curr]
            next_pt = path[idx_next]
            kappa, radius = menger(prev_pt, curr_pt, next_pt)
            tangent = next_pt - curr_pt

            optimal_position = curr_pt
            optimal_velocity = self.get_maximum_turn_velocity(radius, max_a_centr)
            optimal_velocity = np.clip(optimal_velocity, v_target[0], v_target[1])
            optimal_heading = np.arctan2(tangent[1], tangent[0])

            states.append([optimal_position[0], optimal_position[1], optimal_heading, optimal_velocity])

            idx_curr = idx_next
        return np.array(states)

    def get_maximum_turn_velocity(self, r: float, accel_cap: float) -> float:
        return math.sqrt(accel_cap * r)