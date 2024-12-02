import numpy as np

from iiwa_batter import (
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import PITCH_START_POSITION, FLIGHT_TIME_MULTIPLE

from iiwa_batter.swing_simulator import (
    setup_simulator,
    reset_systems,
    run_swing_simulation, 
    parse_simulation_state
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

def find_initial_positions(robot, num_positions, bounding_box=None):
    np.random.seed(0)
    
    robot_constraints = JOINT_CONSTRAINTS[robot]
    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf=robot, robot_constraints=robot_constraints)
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
            np.zeros(NUM_JOINTS),
            PITCH_START_POSITION,
            np.zeros(3),
        )

        if status_dict["result"] == "collision":
            continue
        else:
            # Make sure the sweet spot is not starting in the ground or in front of the ball
            sweet_spot_position = parse_simulation_state(simulator, diagram, "sweet_spot")
            if bounding_box is None:
                if sweet_spot_position[2] < 0:
                    continue
                if sweet_spot_position[0] > 0 and sweet_spot_position[1] < 0:
                    continue
            else:
                if bounding_box["x"][0] > sweet_spot_position[0] or sweet_spot_position[0] > bounding_box["x"][1]:
                    continue
                if bounding_box["y"][0] > sweet_spot_position[1] or sweet_spot_position[1] > bounding_box["y"][1]:
                    continue
                if bounding_box["z"][0] > sweet_spot_position[2] or sweet_spot_position[2] > bounding_box["z"][1]:
                    continue
            handle_position = parse_simulation_state(simulator, diagram, "handle")
            if handle_position[0] > 0:
                continue

        initial_positions.append(candidate_position)

    reset_systems(diagram)
    return initial_positions


def make_trajectory_timesteps(flight_time):
    trajectory_timesteps = np.arange(0, (flight_time*FLIGHT_TIME_MULTIPLE+CONTROL_DT), CONTROL_DT)
    return trajectory_timesteps


def initialize_control_vector(robot_constraints, flight_time):
    if flight_time > 1:
        raise ValueError("Flight time must be less than or equal to 1 second.")
    num_timesteps = len(make_trajectory_timesteps(flight_time))
    control_vector = np.zeros((num_timesteps, NUM_JOINTS))
    for i, torque in enumerate(robot_constraints["torque"].values()):
        control_vector[:, i] = np.random.uniform(-torque, torque, num_timesteps)

    return control_vector


def expand_control_vector(original_control_vector, new_flight_time):
    """Add zeros to the beginning of the control vector to make it longer if necessary."""
    new_trajectory_timesteps = make_trajectory_timesteps(new_flight_time)
    original_length = len(original_control_vector)
    new_length = len(new_trajectory_timesteps)
    zero_fill = np.zeros((new_length-original_length, NUM_JOINTS))
    expanded_control_vector = np.concatenate((zero_fill, original_control_vector), axis=0)

    return expanded_control_vector


def make_torque_trajectory(control_vector, flight_time):
    """Make a torque trajectory from a control vector."""
    trajectory_timesteps = make_trajectory_timesteps(flight_time)
    torque_trajectory = {}
    for i in range(len(trajectory_timesteps)):
        timestep = trajectory_timesteps[i]
        torque_trajectory[timestep] = control_vector[i]
    return torque_trajectory


def perturb_vector(original_vector, variance, upper_limits, lower_limits):
    perturbation = np.random.normal(0, variance, size=original_vector.shape)
    perturbed_vector = np.clip(original_vector + perturbation, lower_limits, upper_limits)
    return perturbed_vector


def descent_step(original_vector, perturbed_vector, original_reward, perturbed_reward, learning_rate, upper_limits, lower_limits):
    """Take a step in the direction of the perturbed vector, scaled by the learning rate.
    
    TODO: I realize that this should be called 'ascent step', because it maximizes reward. Refactor that, eventually.
    """
    # Our vectors are in the form of degrees, degrees/second, and torques. # WARNING! THIS HARD-CODED WHENEVER perturb_vector IS CALLED
    # If the change in reward is too large on a particular step, that will throw the vector off completely.
    # So, clip the change in reward to a reasonable amount.
    effective_reward_difference = np.clip(perturbed_reward - original_reward, -1, 1)

    desired_vector = original_vector + learning_rate * effective_reward_difference * (perturbed_vector - original_vector)
    clipped_vector = np.clip(desired_vector, lower_limits, upper_limits)
    return clipped_vector
