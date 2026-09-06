import numpy as np
import math
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
        self.centerline_pregenerated = np.array([centerline(s) for s in np.linspace(0, track_len, waypoint_count)])
        self.last_centerline_idx = 0

    def search_for_centerline_goalpoint(self, current_x, current_y, lookahead_dist):
        if self.last_centerline_idx > len(self.centerline_pregenerated):
            self.last_centerline_idx = 0
        start_idx = self.last_centerline_idx

        path = self.centerline_pregenerated
        goal_pt = path[self.last_centerline_idx]

        # line-circle intersection, checking each line segment along waypoint list, one at a time
        for i in range(start_idx, len(path) + 10):
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
                        if (idx - self.last_centerline_idx) < len(self.centerline_pregenerated) / 2:
                            self.last_centerline_idx = idx
                            break
                    else:
                        # can't find intersection in next segment, but don't look in this one
                        self.last_centerline_idx = idx + 1
                # no solutions are in range
                else:
                    goal_pt = path[self.last_centerline_idx]
        return goal_pt, self.last_centerline_idx

    def generate_raceline(self):
        pass