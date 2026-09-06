import numpy as np

class PurePursuitController:
    def __init__(self, kdd=0.1, min_lookahead=1.0, max_lookahead=5.0, wheelbase=1.58):
        self.kdd = kdd # look-ahead constant
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = wheelbase

    def get_lookahead(self, v_current):
        return np.clip(self.kdd * v_current, self.min_lookahead, self.max_lookahead)

    def get_steering_output(self, v_current, target_waypoint):
        lookahead_dist = self.get_lookahead(v_current)

        target_x = target_waypoint[0]
        target_y = target_waypoint[1]

        alpha = np.arctan2(target_y, target_x)
        target_steer = np.arctan((2 * self.wheelbase * np.sin(alpha)) / lookahead_dist)

        return target_steer