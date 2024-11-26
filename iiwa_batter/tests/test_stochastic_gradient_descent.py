import numpy as np

from iiwa_batter import (
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    initialize_control_vector,
    expand_control_vector,
    make_trajectory_timesteps,
    make_torque_trajectory,
    descent_step,
    perturb_vector,
)

def test_initialize_control_vector():
    # Ensure the initialized control vector has the correct shape and all the values are within the torque constraints
    np.random.seed(0)

    robot_constraints = JOINT_CONSTRAINTS["iiwa14"]
    time_of_flight = 0.5

    trajectory_timesteps = make_trajectory_timesteps(time_of_flight)
    control_vector = initialize_control_vector(robot_constraints, time_of_flight)

    assert control_vector.shape == (len(trajectory_timesteps), NUM_JOINTS)
    for i in range(NUM_JOINTS):
        assert np.all(control_vector[:, i] <= robot_constraints["torque"][str(i+1)])
        assert np.all(control_vector[:, i] >= -robot_constraints["torque"][str(i+1)])


def test_expand_control_vector():
    np.random.seed(0)
    original_flight_time = 0.05
    original_control_vector = initialize_control_vector(JOINT_CONSTRAINTS["iiwa14"], original_flight_time)

    new_flight_time = 0.08
    expanded_control_vector = expand_control_vector(original_control_vector, new_flight_time)
    expanded_comparison_control_vector = initialize_control_vector(JOINT_CONSTRAINTS["iiwa14"], new_flight_time)

    assert expanded_control_vector.shape == expanded_comparison_control_vector.shape

    for i in range(expanded_control_vector.shape[0]-original_control_vector.shape[0]):
        assert np.all(expanded_control_vector[i] == np.zeros(NUM_JOINTS))
    for i in range(expanded_control_vector.shape[0]-original_control_vector.shape[0], expanded_control_vector.shape[0]):
        assert np.all(expanded_control_vector[i] != np.zeros(NUM_JOINTS)) # Highly unlikely for any to be perfectly 0


def test_make_torque_trajectory():
    # Ensure that the torque trajectory is made correclty
    control_timesteps = make_trajectory_timesteps(0.05)
    control_vector = np.array([
        [0, 0, 0], 
        [1, 1, 1], 
        [2, 2, 2], 
        [3, 3, 3], 
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
    ])

    torque_trajectory = make_torque_trajectory(control_vector, 0.05)

    assert len(torque_trajectory) == len(control_timesteps)
    assert np.all(torque_trajectory[0] == np.array([0, 0, 0]))
    assert np.all(torque_trajectory[0.01] == np.array([1, 1, 1]))
    assert np.all(torque_trajectory[0.04] == np.array([4, 4, 4]))


def test_perturbed_vector_bounds():
    # Ensure the perturbed vector is always within the bounds
    np.random.seed(0)
    original_vector = np.zeros((100, NUM_JOINTS))
    variance = 10
    upper_limits = np.ones(NUM_JOINTS)
    lower_limits = -np.ones(NUM_JOINTS)

    for _ in range(999):
        perturbed_vector = perturb_vector(original_vector, variance, upper_limits, lower_limits)
        for i in range(NUM_JOINTS):
            assert np.all(perturbed_vector[:, i] <= upper_limits[i])
            assert np.all(perturbed_vector[:, i] >= lower_limits[i])


def test_descent_step_bounds():
    # Ensure the descent step is always within the bounds
    original_vector = np.zeros((100, NUM_JOINTS))
    perturbed_vector = np.ones((100, NUM_JOINTS))
    original_reward = 0
    perturbed_reward = 1
    learning_rate = 500
    upper_limits = np.ones(NUM_JOINTS)
    lower_limits = -np.ones(NUM_JOINTS)

    clipped_vector = descent_step(original_vector, perturbed_vector, original_reward, perturbed_reward, learning_rate, upper_limits, lower_limits)

    for i in range(NUM_JOINTS):
        assert np.all(clipped_vector[:, i] <= upper_limits[i])
        assert np.all(clipped_vector[:, i] >= lower_limits[i])
