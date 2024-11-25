import dill
import numpy as np

from iiwa_batter import (
    NUM_JOINTS,
    PACKAGE_ROOT,
    PITCH_DT,
    CONTROL_DT,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    PITCH_START_POSITION,
    FLIGHT_TIME_MULTIPLE,
)
from iiwa_batter.swing_simulator import (
    setup_simulator,
    run_swing_simulation,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.full_trajectory import full_trajectory_reward
from iiwa_batter.swing_optimization.stochastic_gradient_descent import make_torque_trajectory

from iiwa_batter.naive_full_trajectory_optimization import run_naive_full_trajectory_optimization

robot = "iiwa14"
target_velocity_mph = 90
target_position = [0, 0, 0.6]
optimization_name = "take_1"
save_directory = f"{PACKAGE_ROOT}/swing_optimization/trajectories"
test_dt = PITCH_DT

run_naive_full_trajectory_optimization(
    robot=robot,
    target_velocity_mph=target_velocity_mph,
    target_position=target_position,
    optimization_name=optimization_name,
    save_directory=save_directory,
    simulation_dt=test_dt,
    inner_iterations=1,
    outer_iterations=1e4,
    debug_prints=True,
)
