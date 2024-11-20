import json
import numpy as np
from pydrake.geometry import StartMeshcat

from iiwa_batter import PACKAGE_ROOT, CONTACT_DT, NUM_JOINTS
from iiwa_batter.swing_optimization.instantaneous_swing import run_instantaneous_swing
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

# "iiwa14" "kr6r900" "slugger"
robot_constraints = JOINT_CONSTRAINTS["kr6r900"]

# Run gaussian process optimization to find the joint velocity which gives the best ball distance

# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/basic-tour.html

with open(f"{PACKAGE_ROOT}/sandbox/ball_plate_state_90mph.json", "r") as f:
    ball_state = json.load(f)

ball_plate_position = np.zeros(7)
ball_plate_velocity = np.zeros(6)
for key, value in ball_state.items():
    if key == "position":
        for i, val in enumerate(value.items()):
            ball_plate_position[i] = val[1]
    elif key == "velocity":
        for i, val in enumerate(value.items()):
            ball_plate_velocity[i] = val[1]

print(f"Plate ball position: {ball_plate_position[4:]}")
print(f"Plate ball velocity: {ball_plate_velocity[3:]}")
plate_ball_state = (ball_plate_position, ball_plate_velocity)

from bayes_opt import BayesianOptimization
plate_iiwa_position = np.array([0, 1.5, -1.5, 1.6, -0.4, -1.7, 1.6, 0])
parameter_bounds = {
    "joint1": (-robot_constraints["joint_velocity"]["1"], robot_constraints["joint_velocity"]["1"]),
    "joint2": (-robot_constraints["joint_velocity"]["2"], robot_constraints["joint_velocity"]["2"]),
    "joint3": (-robot_constraints["joint_velocity"]["3"], robot_constraints["joint_velocity"]["3"]),
    "joint4": (-robot_constraints["joint_velocity"]["4"], robot_constraints["joint_velocity"]["4"]),
    "joint5": (-robot_constraints["joint_velocity"]["5"], robot_constraints["joint_velocity"]["5"]),
    "joint6": (-robot_constraints["joint_velocity"]["6"], robot_constraints["joint_velocity"]["6"]),
    "joint7": (-robot_constraints["joint_velocity"]["7"], robot_constraints["joint_velocity"]["7"]),
}

def swing_optimization(joint1, joint2, joint3, joint4, joint5, joint6, joint7):
    plate_iiwa_velocity = np.array([joint1, joint2, joint3, joint4, joint5, joint6, joint7])
    return run_instantaneous_swing(None, plate_iiwa_position, plate_iiwa_velocity, plate_ball_state, CONTACT_DT)

optimizer = BayesianOptimization(
    f=swing_optimization,
    pbounds=parameter_bounds,
    random_state=1,
    verbose=1,
)

optimizer.maximize(
    init_points=100,
    n_iter=500,
)

# Get the best results
best_params = optimizer.max

print(f"Best parameters: {best_params}")