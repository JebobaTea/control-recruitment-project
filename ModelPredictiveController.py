import math
import numpy as np

class ModelPredictiveController:
    def __init__(self, wheelbase=1.58, dt: float = 0.1, steering_constraints: tuple[float, float]=(-1.0, 1.0),
                 wheel_constraints: tuple[float, float]=(-0.7, 7)):
        self.wheelbase = wheelbase
        self.dt = 0.1
        self.steering_constraints = steering_constraints
        self.wheel_constraints = wheel_constraints

    def sim_bicycle(self, state, ctrl):
        x, y, phi, v, theta = state
        a, theta_dot = ctrl

        x_new = x + v * self.dt * math.cos(phi)
        y_new= y + v * self.dt * math.sin(phi)
        v_new = v + a * self.dt
        phi_new = phi + self.dt * (v / self.wheelbase) * math.tan(theta)
        theta_new = theta + self.dt * theta_dot

        return np.array([x_new, y_new, v_new, phi_new, theta_new])