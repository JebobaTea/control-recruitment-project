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
    def __init__(self, waypoint_count:int=1000, track_len=419):
        # this particular track repeats after s=419 along centerline
        self.ds = track_len / waypoint_count
        self.centerline_pregenerated = np.array([centerline(s) for s in np.linspace(0, track_len, waypoint_count)])
        self.last_centerline_idx = 0

    def search_for_centerline_goalpoint(self, current_x, current_y, lookahead_dist):
        start_idx = self.last_centerline_idx

        path = self.centerline_pregenerated
        goal_pt = path[self.last_centerline_idx]

        # line-circle intersection, checking each line segment along waypoint list, one at a time
        for i in range(start_idx, len(path) - 1):
            # assumes circle is centered on 0, 0, so correct for offset
            x1 = path[i][0] - current_x
            y1 = path[i][1] - current_y
            x2 = path[i + 1][0] - current_x
            y2 = path[i + 1][1] - current_y
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
                min_x = min(path[i][0], path[i + 1][0])
                min_y = min(path[i][1], path[i + 1][1])
                max_x = max(path[i][0], path[i + 1][0])
                max_y = max(path[i][1], path[i + 1][1])

                # first solution in range
                if (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y):
                    if (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y):
                        # second solution also in range
                        # make the decision by comparing the distance between the intersections and the next point in path
                        # (which intersection closer to second point in path)
                        if pt_dist(sol_pt1, path[i + 1]) < pt_dist(sol_pt2, path[i + 1]):
                            goal_pt = sol_pt1
                        else:
                            goal_pt = sol_pt2
                    else:
                        # only 1st sol in range
                        goal_pt = sol_pt1
                elif (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y):
                    # only second solution in range
                    goal_pt = sol_pt2
                else:
                    # no solutions in range
                    goal_pt = path[self.last_centerline_idx]

                # break if goal point is further along the path
                if pt_dist(goal_pt, path[i + 1]) < pt_dist([current_x, current_y], path[i + 1]):
                    self.last_centerline_idx = i
                    break
                else:
                    # can't find intersection in next segment
                    self.last_centerline_idx = i + 1
        return goal_pt

    def generate_raceline(self):
        pass