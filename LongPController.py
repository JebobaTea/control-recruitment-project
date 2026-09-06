import math
import numpy as np

class LongPController:
    def __init__(self, kp, kc, min_speed, max_speed):
        self.kp = kp
        self.kc = kc
        self.min_speed = min_speed
        self.max_speed = max_speed

    def get_accel_output(self, kappa, current_speed):
        target_speed = self.max_speed * (1 / (kappa * self.kc))
        if current_speed < self.min_speed:
            target_speed = self.max_speed
        d_speed = target_speed - current_speed

        return self.kp * d_speed