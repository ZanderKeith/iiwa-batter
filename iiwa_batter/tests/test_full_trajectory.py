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
    single_full_trajectory_torque_and_position,
)
from iiwa_batter.trajectory_library import (
    LIBRARY_SPEEDS_MPH,
    LIBRARY_POSITIONS,
)

def test_run_single_full_trajectory_torque_only():
    # Try running the trajectory optimization with only the torque values being updated and ensure nothing breaks
    np.random.seed(0)
    robot = "iiwa14"
    robot_constraints = JOINT_CONSTRAINTS[robot]
    torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(LIBRARY_SPEEDS_MPH[0], LIBRARY_POSITIONS[0])
    initial_control_vector = initialize_control_vector(robot_constraints, ball_time_of_flight)

    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf=robot, dt=PITCH_DT, robot_constraints=robot_constraints)
    best_control_vector, _ = single_full_trajectory_torque_only(
        simulator=simulator,
        diagram=diagram,
        initial_joint_positions=np.zeros(NUM_JOINTS),
        original_control_vector=initial_control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
        torque_constraints=torque_constraints,
    )

    for i in range(NUM_JOINTS):
        assert np.all(best_control_vector[:, i] <= torque_constraints[i])
        assert np.all(best_control_vector[:, i] >= -torque_constraints[i])


def test_run_single_full_trajectory_torque_and_position():
    # Try running the trajectory optimization with both the torque and position values being updated and ensure nothing breaks
    np.random.seed(0)
    robot = "iiwa14"
    robot_constraints = JOINT_CONSTRAINTS[robot]
    torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])
    position_constraints_upper = np.array([joint[1] for joint in robot_constraints["joint_range"].values()])
    position_constraints_lower = np.array([joint[0] for joint in robot_constraints["joint_range"].values()])

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(LIBRARY_SPEEDS_MPH[0], LIBRARY_POSITIONS[0])
    initial_joint_position = np.zeros(NUM_JOINTS)
    initial_control_vector = initialize_control_vector(robot_constraints, ball_time_of_flight)

    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf=robot, dt=PITCH_DT, robot_constraints=robot_constraints)
    best_initial_position, best_control_vector, best_reward = single_full_trajectory_torque_and_position(
        simulator=simulator,
        diagram=diagram,
        original_initial_joint_positions=initial_joint_position,
        original_control_vector=initial_control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
        position_constraints_upper=position_constraints_upper,
        position_constraints_lower=position_constraints_lower,
        torque_constraints=torque_constraints,
    )

    assert np.all(best_initial_position <= position_constraints_upper)
    assert np.all(best_initial_position >= position_constraints_lower)
    for i in range(NUM_JOINTS):
        assert np.all(best_control_vector[:, i] <= torque_constraints[i])
        assert np.all(best_control_vector[:, i] >= -torque_constraints[i])
    

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