import numpy as np


def interpolate_trajectory(simulation_timesteps, trajectory):
    """Given the timesteps of the simulation and a torque trajectory, use rectilinear interpolation to have a torque value for each timestep.

    Parameters
    ----------
    simulation_timesteps : np.ndarray
        The timesteps of the simulation.
    trajectory : dict[float, np.ndarray]
        The torque trajectory to interpolate. Keys are times, values are torques.
    """

    original_timesteps = np.array(list(trajectory.keys()))

    interpolated_trajectory = {}
    for i in range(len(trajectory)):
        lower_bound = original_timesteps[i]
        if i < len(trajectory) - 1:
            upper_bound = original_timesteps[i + 1]
        else:
            upper_bound = np.inf  # Last timestep, no upper bound

        intermediate_torques = {
            time: trajectory[lower_bound]
            for time in simulation_timesteps
            if lower_bound <= time < upper_bound
        }

        interpolated_trajectory.update(intermediate_torques)

    interpolated_trajectory.update(
        {original_timesteps[-1]: trajectory[original_timesteps[-1]]}
    )

    return interpolated_trajectory


def make_torque_trajectory(control_vector, num_joints, trajectory_timesteps):
    """Make a torque trajectory from a control vector. First num_joints values are the initial joint positions, the rest are the torques at each timestep."""
    torque_trajectory = {}
    for i in range(len(trajectory_timesteps)):
        timestep = trajectory_timesteps[i]
        torque_trajectory[timestep] = control_vector[
            num_joints * (i + 1) : num_joints * (i + 2)
        ]
    return torque_trajectory
