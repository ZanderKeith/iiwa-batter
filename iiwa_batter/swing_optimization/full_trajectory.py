import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    Diagram,
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import CONTACT_DT, NUM_JOINTS, PITCH_DT
from iiwa_batter.physics import PITCH_START_POSITION, ball_flight_path
from iiwa_batter.sandbox.pitch_check import FLIGHT_TIME_MULTIPLE, make_model_directive

# This actually works somewhat well... I'm surprised it isn't unbearably slow
# This shall be the backup plan in case the more 'intelligently designed' optimization doesn't work


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


def setup_simulator(dt=CONTACT_DT, meshcat=None):
    builder = DiagramBuilder()
    model_directive = make_model_directive(dt)
    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)

    return simulator, station


def reset_simulator(simulator: Simulator):
    context = simulator.get_mutable_context()
    context.SetTime(0)
    simulator.Initialize()


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


def make_torque_trajectory(control_vector, num_joints, trajectory_timesteps):
    """Make a torque trajectory from a control vector. First num_joints values are the initial joint positions, the rest are the torques at each timestep."""
    torque_trajectory = {}
    for i in range(len(trajectory_timesteps)):
        timestep = trajectory_timesteps[i]
        torque_trajectory[timestep] = control_vector[
            num_joints * (i + 1) : num_joints * (i + 2)
        ]
    return torque_trajectory


def stochastic_optimization_full_trajectory(
    simulator: Simulator,
    station: Diagram,
    robot_constraints,
    original_control_vector,
    control_timesteps,
    ball_initial_velocity,
    time_of_flight,
    learning_rate=0.5,
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
    reset_simulator(simulator)
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

    reset_simulator(simulator)
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


def run_full_trajectory(
    meshcat,
    simulator: Simulator,
    station: Diagram,
    initial_joint_positions,
    initial_ball_state_arrays,
    time_of_flight,
    robot_constraints,
    torque_trajectory: dict[float, np.ndarray],
):
    if meshcat is not None:
        meshcat.Delete()

    context = simulator.get_context()
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, [0] * NUM_JOINTS)
    simulator.AdvanceTo(0)
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Set the initial ball state
    ball = plant.GetModelInstanceByName("ball")
    plant.SetPositions(
        plant_context,
        ball,
        np.concatenate([np.ones(1), np.zeros(3), initial_ball_state_arrays[0]]),
    )
    plant.SetVelocities(
        plant_context, ball, np.concatenate([np.zeros(3), initial_ball_state_arrays[1]])
    )

    # Set initial iiwa state
    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(plant_context, iiwa, initial_joint_positions)
    plant.SetVelocities(plant_context, iiwa, [0] * NUM_JOINTS)

    # Make the trajectory

    simulation_duration = time_of_flight * FLIGHT_TIME_MULTIPLE
    timebase = np.arange(0, simulation_duration, PITCH_DT)
    interpolated_trajectory = interpolate_trajectory(timebase, torque_trajectory)

    # Run the pitch
    if meshcat is not None:
        meshcat.StartRecording()

    strike_distance = None
    for t in timebase:
        station.GetInputPort("iiwa.torque").FixValue(
            station_context, interpolated_trajectory[t]
        )
        simulator.AdvanceTo(t)

        # Check for self-collision... eventually
        # station.GetOutputPort("contact_results").Eval(station_context)
        # contact_results = station.GetOutputPort("contact_results")

        # Enforce joint position and velocity limits... eventually
        # joint_positions = plant.GetPositions(plant_context, iiwa)

        # Record the distance between the sweet spot of the bat and the ball at the point when the ball passes through the strike zone
        # to include in our loss function
        if t >= time_of_flight and strike_distance is None:
            plant.HasModelInstanceNamed("sweet_spot")
            sweet_spot = plant.GetModelInstanceByName("sweet_spot")
            sweet_spot_body = plant.GetRigidBodyByName("base", sweet_spot)
            sweet_spot_pose = plant.EvalBodyPoseInWorld(plant_context, sweet_spot_body)
            ball_position = plant.GetPositions(plant_context, ball)[4:]
            strike_distance = np.linalg.norm(
                ball_position - sweet_spot_pose.translation()
            )

    if meshcat is not None:
        meshcat.PublishRecording()

    # Calculate reward
    # If ball position is negative, we missed the ball and should penalize
    # Otherwise, return reward based on the distance the ball travels

    ball_position = plant.GetPositions(plant_context, ball)[4:]

    if ball_position[0] < 0:
        reward = -10 * strike_distance
    else:
        # Determine distance ball travels
        ball_velocity = plant.GetVelocities(plant_context, ball)[3:]
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)  # Absolute distance from origin
        # If the ball is traveling backwards, reward is negative distance
        if land_location[0] < 0:
            reward = -distance
        # If the ball is foul (angle > +/- 45 degrees), reward is half the distance
        elif np.abs(np.arctan(land_location[1] / land_location[0])) > np.pi / 4:
            reward = distance / 2
        # Otherwise, return the distance
        else:
            reward = distance

    return reward
