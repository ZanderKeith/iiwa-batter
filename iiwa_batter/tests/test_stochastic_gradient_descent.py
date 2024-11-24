import numpy as np

from iiwa_batter import CONTROL_DT, PITCH_DT
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
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

