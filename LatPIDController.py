import math
import numpy as np
from collections import deque

class LatPIDController():
    def __init__(self, coeff_config: dict, dt: float = 0.1, error_window=10,
                 steering_constraints: tuple[float, float] = (-1.0, 1.0),
                 wheel_constraints: tuple[float, float] = (-0.7, 7), kdd=0.3,
                 min_lookahead=2.0, max_lookahead=5.0,):
        self.coeff_config = coeff_config
        self._error_buffer = deque(maxlen=error_window)
        self._dt = dt
        self.steering_constraints = steering_constraints
        self.wheel_constraints = wheel_constraints
        self.kdd = kdd  # look-ahead constant
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead

    def get_lookahead(self, v_current):
        return np.clip(self.kdd * v_current, self.min_lookahead, self.max_lookahead)

    def get_steering_input(self, state, target_waypoint):
        x_current, y_current, phi_current, v_current, theta_current = state
        x_target, y_target = target_waypoint

        # operate in global reference frame for simplicity
        wpt_dx = x_target - x_current
        wpt_dy = y_target - y_current
        global_steering_angle = phi_current + theta_current
        vx = v_current * math.cos(global_steering_angle)
        vy = v_current * math.sin(global_steering_angle)

        # signed angle between vectors: aka perp dot product
        # https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
        # atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y)
        # yields minimum angle needed to rotate from a to b,
        # wherein a positive angle is CCW and a negative angle is CW
        # therefore angular error should correspond directly to correct steering input
        angular_error = math.atan2(vx * wpt_dy - vy * wpt_dx, vx * wpt_dx + vy * wpt_dy)

        self._error_buffer.append(angular_error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.get_k_values(v_current=v_current, config=self.coeff_config)

        steer = float(
            np.clip((k_p * angular_error) + (k_d * _de) + (k_i * _ie), self.steering_constraints[0], self.steering_constraints[1])
        )

        return steer, angular_error

    def get_k_values(self, v_current: float, config: dict):
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, k_values in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if v_current < speed_upper_bound:
                k_p, k_d, k_i = k_values["kp"], k_values["kd"], k_values["ki"]
                break
        return np.array([k_p, k_d, k_i])