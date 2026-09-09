import numpy as np
from LatPIDController import LatPIDController
from simulator import Simulator
import WaypointUtility
from PurePursuitController import PurePursuitController
from LongPController import LongPController
from ModelPredictiveController import ModelPredictiveController
from time import process_time

"""

EVAL CONFIG

TF: float (seconds) simulation run duration
CONTROLLER: string (one of the following: PURE_PURSUIT, PID, MPC) which control method to use
            note that only MPC and pure pursuit have valid lap completions

"""
TF = 15
CONTROLLER = "PURE_PURSUIT"


# this particular track repeats after s=104 along centerline
WaypointManager = WaypointUtility.WaypointManager(waypoint_count=200, track_len=104)
WaypointManager.generate_raceline(search_width=(-2, 2))
np.save("wpt_base", WaypointManager.centerline_discrete)
np.save("wpt_opt", WaypointManager.raceline)

sim = Simulator()

v_log = []
t_solves = []
n = 0

def controller(x):
    global n
    global t_solves
    if n % 200 == 0 and n != 0:
        np_t_solves = np.array(t_solves)
        t_avg = np.mean(np_t_solves)
        print("solve progress (pct): ", f"{n / TF : .2f}")
        print("avg MPC solve time per timestep: ", f"{t_avg : .3f}", "s")

    t_solves = []
    n += 1
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    xpos   = x[0]                   # current x position
    ypos   = x[1]                   # current y position
    phi    = np.mod(x[2], 2*np.pi)  # current heading (radians)
    v      = x[3]                   # current velocity
    theta   = x[4]                  # current steering angle
    
    ... # YOUR CODE HERE
    target_waypoint = np.array([0, 0])
    state = [xpos, ypos, phi, v, theta]
    if CONTROLLER == "MPC":
        # admittedly, it'd be cleaner to do some cool packing and unpacking to pass arguments, but we ball
        MPC = ModelPredictiveController(wpt_mgr=WaypointManager, sim=sim, window=5, wheelbase=1.58, dt=0.1,
                                        steering_constraints=(-1.0, 1.0), wheel_constraints=(-0.7, 0.7),
                                        throttle_constraints=(-10.0, 4.0), position_weight=1750.0,
                                        heading_weight=100.0, speed_weight=500.0, effort_weight=10.0, lookahead=2.5,
                                        v_target_range=(0, 12))
        t1 = process_time()
        throttle, steer = MPC.get_inputs(state)
        t2 = process_time()
        t_solves.append(t2 - t1)
    elif CONTROLLER == "PID":
        gain_schedule = {
            "5": {
                "kp": 0.4,
                "kd": 0.2,
                "ki": 0.1
            },
            "10": {
                "kp": 0.3,
                "kd": 0.15,
                "ki": 0.12
            }
        }

        LongitudinalController = LongPController(kp=2, min_speed=4, max_speed=7, throttle_constraints=(-8, 4))
        LateralController = LatPIDController(coeff_config=gain_schedule, dt=0.1, error_window=10,
                                             steering_constraints=(-1.0, 1.0), kdd=2, min_lookahead=3.0,
                                             max_lookahead=6.0)

        target_waypoint, idx = WaypointManager.search_for_goalpoint(xpos, ypos, LateralController.get_lookahead(v))

        steer, error = LateralController.get_steering_input(state, target_waypoint)
        throttle, target_speed = LongitudinalController.get_accel_by_error(error, v)
    elif CONTROLLER == "PURE_PURSUIT":
        LongitudinalController = LongPController(kp=1, kc=5, min_speed=6, max_speed=12, throttle_constraints=(-8, 4))
        LateralController = PurePursuitController(kdd=2, kp=10, min_lookahead=4.0, max_lookahead=6.5,
                                                  wheelbase=1.58, steering_constraints=(-1.0, 1.0))

        target_waypoint, idx = WaypointManager.search_for_goalpoint(xpos, ypos, LateralController.get_lookahead(v))

        steer, kappa, turn_radius = LateralController.get_steering_input(state, target_waypoint)
        throttle, target_speed = LongitudinalController.get_accel_input(kappa, v)
    else:
        print("hey man this is a wendy's we don't sell that here, try a different controller config string")
        raise NotImplementedError

    debug_array = [v, throttle, steer, phi, theta]

    v_log.append(v)
    return np.array([throttle, steer]), target_waypoint, debug_array # too lazy

sim.set_controller(controller)
sim.run(tf=TF)
sim.animate()
sim.plot()
results = sim.get_results()
print(np.count_nonzero(np.array(results[3])))
print(np.count_nonzero(np.array(results[4])))
print(np.mean(np.array(v_log)))