import math
import numpy as np

class LongPController:
    def __init__(self, kp, min_speed, target_speed):
        self.kp = kp
        self.min_speed = min_speed
        self.target_speed = target_speed

    def get_accel_output(self, error, current_speed):
        d_speed = self.target_speed - current_speed
        if current_speed < self.min_speed:
            return self.kp * d_speed
        else:
            return (1 - abs(error)) * self.kp * d_speed