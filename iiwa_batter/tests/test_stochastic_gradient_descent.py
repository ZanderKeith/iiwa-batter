import numpy as np

from iiwa_batter import CONTROL_DT, PITCH_DT
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    descent_step,
    perturb_vector,
)


def test_make_torque_trajectory():
    # Ensure that the torque trajectory is made correclty
    control_timesteps = np.array([0, 1, 2, 3, 4])
    control_vector = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    torque_trajectory = make_torque_trajectory(control_vector, 3, control_timesteps)

    assert len(torque_trajectory) == len(control_timesteps)
    assert np.all(torque_trajectory[0] == np.array([0, 0, 0]))
    assert np.all(torque_trajectory[1] == np.array([1, 1, 1]))
    assert np.all(torque_trajectory[4] == np.array([4, 4, 4]))

def test_perturbed_vector_bounds():
    # Ensure the perturbed vector is always within the bounds
    original_vector = np.array([0, 0, 0])
    variance = 10
    upper_limits = np.array([1, 1, 1])
    lower_limits = np.array([-1, -1, -1])

    for _ in range(100):
        perturbed_vector = perturb_vector(original_vector, variance, upper_limits, lower_limits)
        assert np.all(perturbed_vector <= upper_limits)
        assert np.all(perturbed_vector >= lower_limits)


def test_descent_step_bounds():
    # Ensure the descent step is always within the bounds
    original_vector = np.array([0, 0, 0])
    perturbed_vector = np.array([1, 1, 1])
    original_reward = 0
    perturbed_reward = 1
    learning_rate = 500
    upper_limits = np.array([1, 1, 1])
    lower_limits = np.array([-1, -1, -1])

    clipped_vector = descent_step(original_vector, perturbed_vector, original_reward, perturbed_reward, learning_rate, upper_limits, lower_limits)

    assert np.all(clipped_vector <= upper_limits)
    assert np.all(clipped_vector >= lower_limits)