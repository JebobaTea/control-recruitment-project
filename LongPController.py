import numpy as np

class LongPController:
    def __init__(self, kp=1, kc=1, min_speed=1, max_speed=10, throttle_constraints: tuple[float, float]=(-10, 4)):
        self.kp = kp # throttle gain
        self.kc = kc # how much curvature should affect steering
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.throttle_constraints = throttle_constraints

    def get_accel_input(self, kappa, current_speed):
        target_speed = self.max_speed * (1 / (1 + abs(kappa) * self.kc))
        if current_speed < self.min_speed:
            target_speed = self.max_speed
        d_speed = target_speed - current_speed
        throttle = self.kp * d_speed
        return np.clip(throttle, self.throttle_constraints[0], self.throttle_constraints[1]), target_speed

    def get_accel_by_error(self, error, current_speed):
        target_speed = self.max_speed * (1 / (1 + error))
        if current_speed < self.min_speed:
            target_speed = self.max_speed
        d_speed = target_speed - current_speed
        throttle = self.kp * d_speed
        return np.clip(throttle, self.throttle_constraints[0], self.throttle_constraints[1]), target_speed