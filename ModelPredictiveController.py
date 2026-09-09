import math
import numpy as np
from scipy.optimize import minimize # cvxpy can't handle trigonometric functions, sadge, so scipy instead
from WaypointUtility import WaypointManager, dist_euclid
from simulator import Simulator

class ModelPredictiveController:
    # doesn't this mess just bring tears to your eyes
    def __init__(self, wpt_mgr:WaypointManager, sim: Simulator,window:int=5, wheelbase=1.58, dt: float=0.1,
                 steering_constraints: tuple[float, float]=(-1.0, 1.0),
                 wheel_constraints: tuple[float, float]=(-0.7, 0.7),
                 throttle_constraints: tuple[float, float]=(-10.0, 4.0),
                 position_weight: float=1500.0, heading_weight: float=100.0,
                 speed_weight: float=500.0, effort_weight: float=10.0, lookahead: float=2.5,
                 v_target_range: tuple[float, float]=(0, 12)):
        self.wheelbase = wheelbase
        self.dt = dt
        self.steering_constraints = steering_constraints
        self.wheel_constraints = wheel_constraints
        self.throttle_constraints = throttle_constraints
        self.WaypointManager = wpt_mgr
        self.Simulator = sim
        self.window = window
        self.lookahead = lookahead
        self.position_weight = position_weight
        self.heading_weight = heading_weight
        self.speed_weight = speed_weight
        self.effort_weight = effort_weight
        self.v_target_range = v_target_range

    def _sim_bicycle(self, state, ctrl):
        x_current, y_current, phi_current, v_current, theta_current = state
        a, theta_dot = ctrl

        x_new = x_current + v_current * self.dt * math.cos(phi_current)
        y_new= y_current + v_current * self.dt * math.sin(phi_current)
        v_new = v_current + a * self.dt
        phi_new = phi_current + self.dt * (v_current / self.wheelbase) * math.tan(theta_current)
        theta_new = theta_current + self.dt * theta_dot

        return np.array([x_new, y_new, phi_new, v_new, theta_new])

    def _cost(self, state, ctrl):
        x_current, y_current, phi_current, v_current, theta_current = state
        references = self.WaypointManager.get_optimal_states(x_current, y_current, lookahead_dist=self.lookahead, window=self.window, v_target=self.v_target_range)

        # ctrl is in the shape of a 1d array, wherein the inputs come in format
        # [a_1, theta_dot_1, a_2, theta_dot_2 ... a_n, theta_dot_n]
        ctrl = ctrl.reshape(self.window, 2)

        # state array argument is mutable
        state_new = state[:]
        cost = 0
        for idx, ref_this_frame in enumerate(references):
            ctrl_this_frame = ctrl[idx]
            state_new = self._sim_bicycle(state_new, ctrl_this_frame)
            x_new, y_new, phi_new, v_new, theta_new = state_new
            x_target, y_target, phi_target, v_target = ref_this_frame

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

    def _breaks_constraints(self, state, ctrl):
        ctrl = ctrl.reshape(self.window, 2)
        # state array argument is mutable
        state_new = state[:]
        res = []
        for ctrl_this_frame in ctrl:
            state_new = self._sim_bicycle(state_new, ctrl_this_frame)
            x_new, y_new, phi_new, v_new, theta_new = state_new
            # easier and less time consuming than importing cones and doing hitbox check, so cheeky hack fix
            if self.Simulator._check_collision(state_new):
                res.append(-1)
            else:
                res.append(1)
            # change from previous 1 vs -1 approach:
            # optimizer throws a hissy fit if the constraint returns are on/off
            # instead of smooth and continuous
            new_accel = self.Simulator._get_accel(state_new, ctrl_this_frame)
            res.append(11 - new_accel) # constraints should be padded conservatively
            res.append(-(self.wheel_constraints[0] - theta_new))
            res.append(self.wheel_constraints[1] - theta_new)
        return np.array(res) # inequality operator functions on entire np arrays

    def get_inputs(self, state):
        initial_guess = np.zeros(self.window * 2)

        # cost function input is in format [a_1, theta_dot_1, a_2, theta_dot_2 ... a_n, theta_dot_n]
        # so bounds must be [(a_min, a_max), (theta_min, theta_max), (a_min, a_max) ... ...]
        bounds = []
        for i in range(self.window):
            bounds.append(self.throttle_constraints)
            bounds.append(self.steering_constraints)

        # ineq satisfied when result is non-negative
        constraints = {
            "type": "ineq",
            "fun": (lambda y: self._breaks_constraints(state, y))
        }

        inp = minimize(fun=lambda x: self._cost(state, x), x0=initial_guess, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 200})
        if inp.success:
            inp_optimal = inp.x.reshape(self.window, 2)
            return inp_optimal[0]
        else:
            return [-2, 0]
