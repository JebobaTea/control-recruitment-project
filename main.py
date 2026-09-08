import numpy as np
from simulator import Simulator, centerline
import WaypointUtility
from PurePursuitController import PurePursuitController
from LongPController import LongPController
from ModelPredictiveController import ModelPredictiveController

# this particular track repeats after s=104 along centerline
WaypointManager = WaypointUtility.WaypointManager(waypoint_count=200, track_len=104)
WaypointManager.generate_raceline(search_width=(-2, 2))
np.save("wpt_base", WaypointManager.centerline_discrete)
np.save("wpt_opt", WaypointManager.raceline)

sim = Simulator()

v_log = []
tf = 15
n = 0

def controller(x):
    global n
    if n % 100 == 0:
        print("solve progress (pct): ", f"{n / tf : .2f}")
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
    #LateralController = PurePursuitController(kdd=2, kp=10, min_lookahead=4.0, max_lookahead=6.5, wheelbase=1.58, steering_constraints=(-1.0, 1.0))
    #LongitudinalController = LongPController(kp=1, kc=5, min_speed=6, max_speed=12, throttle_constraints=(-8, 4))
    #target_waypoint, idx = WaypointManager.search_for_goalpoint(xpos, ypos, LateralController.get_lookahead(v))

    state = [xpos, ypos, phi, v, theta]
    #steer, kappa, turn_radius = LateralController.get_steering_input(state, target_waypoint)
    #throttle, target_speed = LongitudinalController.get_accel_input(kappa, v)
    MPC = ModelPredictiveController(wpt_mgr=WaypointManager, sim=sim, window=5, wheelbase=1.58, dt=0.1,
                                    steering_constraints=(-1.0, 1.0), wheel_constraints=(-0.7, 0.7),
                                    throttle_constraints=(-10.0, 4.0))
    throttle, steer = MPC.get_inputs(state)

    debug_array = [v, throttle, steer, phi, theta]

    v_log.append(v)
    return np.array([throttle, steer]), np.array([0, 0]), debug_array # too lazy

sim.set_controller(controller)
sim.run(tf=tf)
sim.animate()
sim.plot()
results = sim.get_results()
print(np.count_nonzero(np.array(results[3])))
print(np.count_nonzero(np.array(results[4])))
print(np.mean(np.array(v_log)))