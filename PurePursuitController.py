import math
import numpy as np

class PurePursuitController:
    def __init__(self, kdd=0.3, kp=10, min_lookahead=2.0, max_lookahead=5.0,
                 wheelbase=1.58, steering_constraints: tuple[float, float]=(-1.0, 1.0),
                 wheel_constraints: tuple[float, float]=(-0.7, 7)):
        self.kdd = kdd # look-ahead constant
        self.kp = kp # steering gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = wheelbase
        self.steering_constraints = steering_constraints
        self.wheel_constraints = wheel_constraints

    def get_lookahead(self, v_current):
        return np.clip(self.kdd * v_current, self.min_lookahead, self.max_lookahead)

    def get_steering_input(self, v_current, target_waypoint, current_x, current_y, theta, phi):
        lookahead_dist = self.get_lookahead(v_current)

        target_x = target_waypoint[0]
        target_y = target_waypoint[1]

        compensated_target_x = target_x - current_x
        compensated_target_y = target_y - current_y

        # now working with relative distance, but angle is still absolute
        relative_target_x = np.cos(-phi) * compensated_target_x - np.sin(-phi) * compensated_target_y
        relative_target_y = np.sin(-phi) * compensated_target_x + np.cos(-phi) * compensated_target_y

        # https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
        alpha = np.arctan2(relative_target_y, relative_target_x)
        turn_radius = lookahead_dist / (2 * np.sin(alpha))
        kappa = 1 / turn_radius
        target_steer = np.arctan((2 * self.wheelbase * np.sin(alpha)) / lookahead_dist)

        # current steering angle vs desired steering angle at this very instant
        steering_error = target_steer - theta
        # smallest difference between two angles
        if steering_error < -math.pi: steering_error += math.pi
        elif steering_error > math.pi: steering_error -= math.pi

        steering_input = steering_error * self.kp

        return np.clip(steering_input, self.steering_constraints[0], self.steering_constraints[1]), kappa, turn_radius