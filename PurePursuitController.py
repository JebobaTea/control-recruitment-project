import math
import numpy as np

class PurePursuitController:
    def __init__(self, kdd=0.3, min_lookahead=2.0, max_lookahead=5.0, wheelbase=1.58):
        self.kdd = kdd # look-ahead constant
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = wheelbase

    def get_lookahead(self, v_current):
        return np.clip(self.kdd * v_current, self.min_lookahead, self.max_lookahead)

    def get_steering_output(self, v_current, target_waypoint, current_x, current_y, theta, phi):
        lookahead_dist = self.get_lookahead(v_current)

        target_x = target_waypoint[0]
        target_y = target_waypoint[1]

        compensated_target_x = target_x - current_x
        compensated_target_y = target_y - current_y

        # now working with relative distance, but angle is still absolute
        relative_target_x = np.cos(-phi) * compensated_target_x - np.sin(-phi) * compensated_target_y
        relative_target_y = np.sin(-phi) * compensated_target_x + np.cos(-phi) * compensated_target_y

        alpha = np.arctan2(relative_target_y, relative_target_x)
        turn_radius = lookahead_dist / (2 * np.sin(alpha))
        kappa = 1 / turn_radius
        target_steer = np.arctan((2 * self.wheelbase * np.sin(alpha)) / lookahead_dist)

        # current steering angle vs desired steering angle at this very instant
        steering_error = target_steer - theta
        # smallest difference between two angles
        if steering_error < -math.pi: steering_error += math.pi
        elif steering_error > math.pi: steering_error -= math.pi

        return steering_error * 10, kappa, turn_radius