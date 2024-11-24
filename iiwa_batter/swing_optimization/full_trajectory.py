from iiwa_batter.physics import PITCH_START_POSITION

# This actually works somewhat well... I'm surprised it isn't unbearably slow
# This shall be the backup plan in case the more 'intelligently designed' optimization doesn't work


def initialize_control_vector(robot_constraints, num_timesteps):
    # First index is the initial position
    # All the next ones are the control torques
    num_joints = len(robot_constraints["torque"])
    control_vector = np.zeros(num_joints + num_timesteps * num_joints)

    # Set the initial position
    for i, joint in enumerate(robot_constraints["joint_range"].values()):
        control_vector[i] = np.random.uniform(joint[0], joint[1])

    for t in range(num_timesteps):
        for i, torque in enumerate(robot_constraints["torque"].values()):
            control_vector[num_joints + t * num_joints + i] = np.random.uniform(
                -torque, torque
            )

    return control_vector


def stochastic_optimization_full_trajectory(
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
