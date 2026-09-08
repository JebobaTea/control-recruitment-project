import math
import numpy as np
from scipy.optimize import minimize # cvxpy can't handle trigonometric functions, sadge, so scipy instead
from WaypointUtility import WaypointManager, dist_euclid, pt_dist

class ModelPredictiveController:
    def __init__(self, wpt_mgr:WaypointManager, window:int=5, wheelbase=1.58, dt: float=0.1,
                 steering_constraints: tuple[float, float]=(-1.0, 1.0),
                 wheel_constraints: tuple[float, float]=(-0.7, 7),
                 throttle_constraints: tuple[float, float]=(-10.0, 4.0)):
        self.wheelbase = wheelbase
        self.dt = dt
        self.steering_constraints = steering_constraints
        self.wheel_constraints = wheel_constraints
        self.throttle_constraints = throttle_constraints
        self.WaypointManager = wpt_mgr
        self.window = window

        # magic values, for now
        self.temp_magic_lookahead = 5.0
        self.position_weight = 10.0
        self.heading_weight = 100.0
        self.speed_weight = 100.0
        self.effort_weight = 1.0

    def _sim_bicycle(self, state, ctrl):
        x_current, y_current, phi_current, v_current, theta_current = state
        a, theta_dot = ctrl

        x_new = x_current + v_current * self.dt * math.cos(phi_current)
        y_new= y_current + v_current * self.dt * math.sin(phi_current)
        v_new = v_current + a * self.dt
        phi_new = phi_current + self.dt * (v_current / self.wheelbase) * math.tan(theta_current)
        theta_new = theta_current + self.dt * theta_dot

        return np.array([x_new, y_new, v_new, phi_new, theta_new])

    def _cost(self, state, ctrl):
        x_current, y_current, phi_current, v_current, theta_current = state
        references = self.WaypointManager.get_optimal_states(x_current, y_current, self.temp_magic_lookahead, self.window)
        # ctrl is in the shape of a 1d array, wherein the inputs come in format
        # [a_1, theta_dot_1, a_2, theta_dot_2 ... a_n, theta_dot_n]
        ctrl = ctrl.reshape(self.window, 2)

        # haha redundancy but new var name helps with code tracing in my head soooooo
        # did you know pycharm linting yells at you for not punctuating interjections
        state_new = state
        cost = 0
        for idx, ref in enumerate(references):
            ctrl_this_frame = ctrl[idx]
            state_new = self._sim_bicycle(state_new, ctrl_this_frame)
            x_new, y_new, phi_new, v_new, theta_new = state_new
            x_target, y_target, phi_target, v_target = ref

            pos_error = dist_euclid(x_new, y_new, x_target, y_target)
            cost += self.position_weight * pos_error ** 2

            head_error = phi_new - phi_target
            if head_error < -math.pi: head_error += math.pi
            elif head_error > math.pi: head_error -= math.pi
            cost += self.heading_weight * head_error ** 2

            v_error = v_new - v_target
            cost += self.speed_weight * v_error ** 2

            effort = ctrl_this_frame[0] ** 2 + ctrl_this_frame[1] ** 2
            cost += effort

        return cost

    def get_inputs(self, state):
        # why is there no e in np.zeros
        initial_guess = np.zeros(self.window * 2)

        # cost function input is in format [a_1, theta_dot_1, a_2, theta_dot_2 ... a_n, theta_dot_n]
        # so bounds must be [(a_min, a_max), (theta_min, theta_max), (a_min, a_max) ... ...]
        bounds = []
        for i in range(self.window):
            bounds.append(self.throttle_constraints)
            bounds.append(self.steering_constraints)

        inp = minimize(fun=lambda x: self._cost(state, x), x0=initial_guess, bounds=bounds)
        if inp.success:
            inp_optimal = inp.x.reshape(self.window, 2)
            return inp_optimal[0]
        else:
            return ([0, 0])
