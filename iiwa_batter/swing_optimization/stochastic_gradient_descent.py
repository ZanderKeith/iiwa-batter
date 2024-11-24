import numpy as np

def make_torque_trajectory(control_vector, num_joints, trajectory_timesteps):
    """Make a torque trajectory from a control vector. First num_joints values are the initial joint positions, the rest are the torques at each timestep."""
    torque_trajectory = {}
    for i in range(len(trajectory_timesteps)):
        timestep = trajectory_timesteps[i]
        torque_trajectory[timestep] = control_vector[
            num_joints * i: num_joints * (i + 1)
        ]
    return torque_trajectory

def descent_step(original_vector, perturbed_vector, original_reward, perturbed_reward, learning_rate, upper_limits, lower_limits):
    """Take a step in the direction of the perturbed vector, scaled by the learning rate."""
    desired_vector = original_vector + learning_rate * (perturbed_reward - original_reward) * (perturbed_vector - original_vector)
    clipped_vector = np.clip(desired_vector, lower_limits, upper_limits)
    return clipped_vector
    
def perturb_vector(original_vector, variance, upper_limits, lower_limits):
    perturbation = np.random.normal(0, variance, size=original_vector.shape)
    perturbed_vector = np.clip(original_vector + perturbation, lower_limits, upper_limits)
    return perturbed_vector