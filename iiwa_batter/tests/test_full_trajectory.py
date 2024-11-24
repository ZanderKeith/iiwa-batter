import numpy as np


from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

from iiwa_batter.swing_optimization.full_trajectory import (
    single_full_trajectory,
    multi_full_trajectory,
)

def test_check_valid_initial_position():

    valid_initial_position = np.array([0, 0, 0, 0, 0, 0, 0])
    invalid_initial_position = np.array([0, 1, 1, 1, 1, 0, 1])

    #assert 
    pass

def test_run_single_full_trajectory():
    pass

# def test_run_multi_full_trajectory():
#     # Try running the full trajectory optimization
#     np.random.seed(0)

#     robot = "iiwa14"
#     robot_constraints = JOINT_CONSTRAINTS[robot]

#     pitch_speeds_mph = [90]
#     target_positions = [[0, 0, 0.6]]

#     targets = []
#     for speed in pitch_speeds_mph:
#         for target in target_positions:
#             targets.append({"pitch_speed_mph": speed, "target_position": target})

#     original_initial_position = np.array([0, 0, 0, 0, 0, 0, 0])

#     best_initial_position, multi_full_trajectory(
#         targets,
#         robot_constraints,
#         original_initial_position)