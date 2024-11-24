import numpy as np

from iiwa_batter import (
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import PITCH_START_POSITION

from iiwa_batter.swing_simulator import reset_systems, run_swing_simulation, parse_simulation_state

def find_initial_positions(simulator, diagram, robot_constraints, num_positions):
    initial_positions = []
    while len(initial_positions) < num_positions:
        # Generate a random initial position
        candidate_position = np.zeros(NUM_JOINTS)
        for i, joint in enumerate(robot_constraints["joint_range"].values()):
            joint_position = np.random.uniform(joint[0], joint[1])
            candidate_position[i] = joint_position

        # Check if the initial position is valid by determining if there is no self-contact after
        # only applying gravity for a short period of time.
        reset_systems(diagram, new_torque_trajectory={})
        status_dict = run_swing_simulation(
            simulator,
            diagram,
            0,
            CONTROL_DT, # Giving it a full control timestep on its own
            candidate_position,
            np.array([0]*NUM_JOINTS),
            PITCH_START_POSITION,
            np.zeros(3),
        )

        if status_dict["result"] == "collision":
            continue
        else:
            # Make sure the sweet spot is not starting in the ground or in front of the ball
            sweet_spot_position = parse_simulation_state(simulator, diagram, "sweet_spot")
            if sweet_spot_position[2] < 0:
                continue
            if sweet_spot_position[0] > 0 and sweet_spot_position[1] < 1:
                continue

        initial_positions.append(candidate_position)


def initialize_control_vector(robot_constraints, num_timesteps):
    control_vector = np.zeros(num_timesteps * NUM_JOINTS)

    for t in range(num_timesteps):
        for i, torque in enumerate(robot_constraints["torque"].values()):
            control_vector[t * NUM_JOINTS + i] = np.random.uniform(-torque, torque)

    return control_vector


def make_torque_trajectory(control_vector, num_joints, trajectory_timesteps):
    """Make a torque trajectory from a control vector. First num_joints values are the initial joint positions, the rest are the torques at each timestep."""
    torque_trajectory = {}
    for i in range(len(trajectory_timesteps)):
        timestep = trajectory_timesteps[i]
        torque_trajectory[timestep] = control_vector[
            num_joints * i: num_joints * (i + 1)
        ]
    return torque_trajectory


def perturb_vector(original_vector, variance, upper_limits, lower_limits):
    perturbation = np.random.normal(0, variance, size=original_vector.shape)
    perturbed_vector = np.clip(original_vector + perturbation, lower_limits, upper_limits)
    return perturbed_vector


def descent_step(original_vector, perturbed_vector, original_reward, perturbed_reward, learning_rate, upper_limits, lower_limits):
    """Take a step in the direction of the perturbed vector, scaled by the learning rate."""
    desired_vector = original_vector + learning_rate * (perturbed_reward - original_reward) * (perturbed_vector - original_vector)
    clipped_vector = np.clip(desired_vector, lower_limits, upper_limits)
    return clipped_vector
    
