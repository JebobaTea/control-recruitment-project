import math

import numpy as np
from simulator import Simulator, centerline
import WaypointUtility
from PurePursuitController import PurePursuitController
from LongPController import LongPController

sim = Simulator()
WaypointManager = WaypointUtility.WaypointManager(waypoint_count=500, track_len=105)
# this particular track repeats after s=105 along centerline

tpt_log = []

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
    LateralController = PurePursuitController(kdd=1, min_lookahead=3.0, max_lookahead=6.0, wheelbase=1.58)
    LongitudinalController = LongPController(2.5, 3, 12)
    target_waypoint, idx = WaypointManager.search_for_centerline_goalpoint(xpos, ypos, LateralController.get_lookahead(v))
    steer, target = LateralController.get_steering_output(v, target_waypoint, xpos, ypos, theta, phi)
    steer = np.clip(steer, -1.0, 1.0)

    throttle = LongitudinalController.get_accel_output(steer, v)
    throttle = np.clip(throttle, -10, 4)

    debug_array = [steer, target, theta, phi]
    debug_array = [x * 180 / math.pi for x in debug_array]
    debug_array.append(idx)
    return np.array([throttle, steer]), target_waypoint, debug_array
sim.set_controller(controller)
sim.run()
#sim.animate()
sim.plot()
results = sim.get_results()
print(np.count_nonzero(np.array(results[3])))
print(np.count_nonzero(np.array(results[4])))
