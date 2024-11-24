import numpy as np

from iiwa_batter import (
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import (
    setup_simulator,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    initialize_control_vector,
)
from iiwa_batter.swing_optimization.full_trajectory import (
    single_full_trajectory_torque_only,
    multi_full_trajectory,
)

def test_run_single_full_trajectory_torque_only():
    # Try running the trajectory optimization with only the torque values being updated and ensure nothing breaks
    np.random.seed(0)
    robot_constraints = JOINT_CONSTRAINTS["iiwa14"]
    torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(90, [0, 0, 0.6])
    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    initial_control_vector = initialize_control_vector(robot_constraints, len(trajectory_timesteps))

    simulator, diagram = setup_simulator(torque_trajectory={}, dt=PITCH_DT, robot_constraints=robot_constraints)
    best_control_vector, best_reward = single_full_trajectory_torque_only(
        simulator=simulator,
        diagram=diagram,
        initial_joint_positions=np.array([0]*NUM_JOINTS),
        original_control_vector=initial_control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
        torque_constraints=torque_constraints,
    )

    for i in range(NUM_JOINTS):
        assert np.all(best_control_vector[:, i] <= torque_constraints[i])
        assert np.all(best_control_vector[:, i] >= -torque_constraints[i])


def test_run_single_full_trajectory_torque_and_position():
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