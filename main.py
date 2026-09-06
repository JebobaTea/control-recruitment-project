import numpy as np
from simulator import Simulator, centerline
import WaypointUtility
from PurePursuitController import PurePursuitController

sim = Simulator()
WaypointManager = WaypointUtility.WaypointManager(waypoint_count=1000, track_len=419)

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
    LateralController = PurePursuitController(kdd=0.3, min_lookahead=2.0, max_lookahead=50.0, wheelbase=1.58)
    target_waypoint = WaypointManager.search_for_centerline_goalpoint(xpos, ypos, LateralController.get_lookahead(v))
    steer = LateralController.get_steering_output(v, target_waypoint)

    print(f"{xpos:2f}", f"{ypos:2f}")
    print(target_waypoint)
    print(theta)
    print()
    if v > 50:
        a = 0
    else:
        a = 5

    return np.array([a, steer]), target_waypoint

sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()