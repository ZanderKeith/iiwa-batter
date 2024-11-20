import numpy as np

from iiwa_batter.swing_optimization.full_trajectory import (
    interpolate_trajectory,
    make_torque_trajectory,
)
from iiwa_batter import PITCH_TIMESTEP, CONTROL_TIMESTEP


def test_make_torque_trajectory():
    # Ensure that the torque trajectory is made correclty
    control_timesteps = np.array([0, 1, 2, 3, 4])
    control_vector = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    torque_trajectory = make_torque_trajectory(control_vector, 3, control_timesteps)

    assert len(torque_trajectory) == len(control_timesteps)
    assert np.all(torque_trajectory[0] == np.array([0, 0, 0]))
    assert np.all(torque_trajectory[1] == np.array([1, 1, 1]))
    assert np.all(torque_trajectory[4] == np.array([4, 4, 4]))


def test_interpolate_trajectory_coarse():
    control_timesteps = np.array([0, 1, 2, 3, 4])
    control_vector = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    torque_trajectory = make_torque_trajectory(control_vector, 3, control_timesteps)

    new_timesteps = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    new_trajectory = interpolate_trajectory(new_timesteps, torque_trajectory)

    assert len(new_trajectory) == len(new_timesteps)

    assert np.all(new_trajectory[0] == np.array([0, 0, 0]))
    assert np.all(new_trajectory[0.5] == np.array([0, 0, 0]))
    assert np.all(new_trajectory[1] == np.array([1, 1, 1]))
    assert np.all(new_trajectory[1.5] == np.array([1, 1, 1]))
    assert np.all(new_trajectory[4] == np.array([4, 4, 4]))

def test_interpolate_trajectory_fine():
    control_timesteps = np.arange(0, 0.1, CONTROL_TIMESTEP)
    simulation_timesteps = np.arange(0, 0.1, PITCH_TIMESTEP)

    control_vector = np.concatenate([np.array([-1]), range(len(control_timesteps))])
    original_trajectory = make_torque_trajectory(control_vector, 1, control_timesteps)
    new_trajectory = interpolate_trajectory(simulation_timesteps, original_trajectory)

    assert len(new_trajectory) == len(simulation_timesteps)

    for i in range(len(control_timesteps)-1):
        lower_bound = control_timesteps[i]
        upper_bound = control_timesteps[i + 1]

        intermediate_torques = [new_trajectory[time] for time in new_trajectory.keys() if lower_bound <= time < upper_bound]

        assert np.all(np.array(intermediate_torques) == original_trajectory[lower_bound])
