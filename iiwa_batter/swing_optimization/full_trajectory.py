import numpy as np
from multiprocessing import Pool

from pydrake.all import (
    Diagram,
    Simulator,
)
from iiwa_batter import (
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
    PITCH_DT,
)

from iiwa_batter.physics import (
    PITCH_START_POSITION,
    FLIGHT_TIME_MULTIPLE,
    find_ball_initial_velocity,
    ball_flight_path,
    ball_distance_multiplier,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    reset_systems,
    run_swing_simulation,
    setup_simulator,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import make_torque_trajectory

# This actually works somewhat well... I'm surprised it isn't unbearably slow
# This shall be the backup plan in case the more 'intelligently designed' optimization doesn't work

def full_trajectory_reward(
    simulator: Simulator,
    diagram: Diagram,
    initial_joint_positions,
    control_vector,
    ball_initial_velocity,
    ball_time_of_flight,
):

    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    torque_trajectory = make_torque_trajectory(control_vector, NUM_JOINTS, trajectory_timesteps)
    reset_systems(diagram, torque_trajectory)

    status_dict = run_swing_simulation(
        simulator,
        diagram,
        start_time=0,
        end_time=ball_time_of_flight*FLIGHT_TIME_MULTIPLE,
        initial_joint_positions=initial_joint_positions,
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_velocity=ball_initial_velocity,
    )

    result = status_dict["result"]
    if result == "contact":
        return -10 * status_dict["contact_severity"]
    elif result == "miss":
        return -10 * status_dict["closest_approach"]
    elif result == "hit":
        ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)
        multiplier = ball_distance_multiplier(land_location)
        return distance * multiplier

def single_full_trajectory(
    simulator: Simulator,
    station: Diagram,
    robot_constraints,
    original_control_vector,
    control_timesteps,
    ball_initial_velocity,
    time_of_flight,
    learning_rate=1,
):
    """Run stochastic optimization to find the best control vector for the full swing trajectory.

    Parameters
    ----------
    simulator : Simulator
        The simulator to run the trajectory on. Already initialized with setup_simulator.
    station : Diagram
        The station to run the trajectory on.
    robot_constraints : dict
        The constraints for the robot.
    original_control_vector : np.ndarray
        The original control vector to optimize.
    control_timesteps : np.ndarray
        The timesteps for the control vector.
    ball_initial_velocity : np.ndarray
        The initial velocity of the ball.
    time_of_flight : float
        Time of flight for the ball from the pitch to the strike zone.
    learning_rate : float, optional
        The learning rate for the optimization, by default 0.01.

    Returns
    -------
    updated_control_vector : np.ndarray
        Control vector that has been moved in the direction of the gradient.
    original_reward : float
        Reward from the simulation with the original control vector.
    reward_difference: float
        Difference in reward between the original and perturbed control vectors. If perturbed was better, will be positive.
    """

    position_variance = np.deg2rad(1)
    torque_variance = 1  # About 1% of the max torque

    num_joints = len(robot_constraints["torque"])

    # Determine the loss from this control vector
    torque_trajectory = make_torque_trajectory(
        original_control_vector, num_joints, control_timesteps
    )
    reset_systems(simulator)
    original_reward = run_full_trajectory(
        None,
        simulator,
        station,
        original_control_vector[:num_joints],
        [PITCH_START_POSITION, ball_initial_velocity],
        time_of_flight,
        robot_constraints,
        torque_trajectory,
    )

    # Perturb the control vector, ensuring that the joint constraints are still satisfied
    perturbed_vector = np.empty_like(original_control_vector)
    for i, joint in enumerate(robot_constraints["joint_range"].values()):
        perturbation = np.random.normal(0, position_variance)
        capped_perturbation = np.clip(
            original_control_vector[i] + perturbation, joint[0], joint[1]
        )
        perturbed_vector[i] = capped_perturbation - original_control_vector[i]

    for t in range(len(control_timesteps)):
        for i, torque in enumerate(robot_constraints["torque"].values()):
            perturbation = np.random.normal(0, torque_variance)
            capped_perturbation = np.clip(
                original_control_vector[num_joints + t * num_joints + i] + perturbation,
                -torque,
                torque,
            )
            perturbed_vector[num_joints + t * num_joints + i] = (
                capped_perturbation
                - original_control_vector[num_joints + t * num_joints + i]
            )

    perturbed_control_vector = original_control_vector + perturbed_vector
    perturbed_torque_trajectory = make_torque_trajectory(
        perturbed_control_vector, num_joints, control_timesteps
    )

    reset_systems(simulator)
    perturbed_reward = run_full_trajectory(
        None,
        simulator,
        station,
        perturbed_control_vector[:num_joints],
        [PITCH_START_POSITION, ball_initial_velocity],
        time_of_flight,
        robot_constraints,
        perturbed_torque_trajectory,
    )

    updated_control_vector = (
        original_control_vector
        + learning_rate * (perturbed_reward - original_reward) * perturbed_vector
    )

    reward_difference = perturbed_reward - original_reward

    return updated_control_vector, original_reward, reward_difference

def multi_full_trajectory(
    targets,
    robot_constraints,
    original_initial_position,
    simulation_dt=CONTACT_DT,
    substeps=8, # How many single runs to do between updating the initial position
    save_interval=100,
    top_level_iterations=1000, # How many times to update the initial position
):
    # Initialize the simulator and diagram
    # Keeping these separate for each trajectory, can't re-make them over and over or it causes memory issues
    target_details = {}
    for i, target in enumerate(targets):
        # Determine how many timesteps are needed for the control vector
        simulator, diagram = setup_simulator(
            torque_trajectory={},
            dt = simulation_dt,
            robot_constraints=robot_constraints,
            model_urdf=robot_constraints["model"]
        )
        target_dict = {
            "simulator": simulator,
            "diagram": diagram,
            "target": target,
            "substeps": substeps,
            "initial_position": original_initial_position,
        }
        target_details[i] = target_dict

    best_reward = -np.inf
    best_initial_position = original_initial_position
    for i in range(top_level_iterations):
        pass

        # Determine the loss using the current initial position

        # Determine the loss after after perturbing the initial position

        # Update the initial position based on the gradient