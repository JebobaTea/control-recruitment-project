import math

import numpy as np
from simulator import Simulator, centerline
import WaypointUtility
from PurePursuitController import PurePursuitController
from LongPController import LongPController

sim = Simulator()
WaypointManager = WaypointUtility.WaypointManager(waypoint_count=500, track_len=105)
# this particular track repeats after s=105 along centerline

v_log = []

def controller(x):
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
    v_log.append(v)
    LateralController = PurePursuitController(kdd=2, min_lookahead=4.0, max_lookahead=6.5, wheelbase=1.58)
    LongitudinalController = LongPController(1, 10, 6, 12)
    target_waypoint, idx = WaypointManager.search_for_centerline_goalpoint(xpos, ypos, LateralController.get_lookahead(v))
    steer, kappa, turn_radius = LateralController.get_steering_output(v, target_waypoint, xpos, ypos, theta, phi)
    steer_clipped = np.clip(steer, -1.0, 1.0)

    throttle, target_speed = LongitudinalController.get_accel_output(kappa, v)
    throttle_clipped = np.clip(throttle, -8, 4)

    debug_array = [v, target_speed, kappa, throttle_clipped]
    return np.array([throttle_clipped, steer_clipped]), target_waypoint, debug_array

sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()
results = sim.get_results()
print(np.count_nonzero(np.array(results[3])))
print(np.count_nonzero(np.array(results[4])))
print(np.mean(np.array(v_log)))