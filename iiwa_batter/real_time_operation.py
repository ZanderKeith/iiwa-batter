import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    PITCH_START_POSITION,
)
from iiwa_batter.swing_simulator import (
    setup_simulator,
    reset_systems,
    run_swing_simulation,
    parse_simulation_state,
)
from iiwa_batter.swing_optimization.partial_trajectory import partial_trajectory_reward
from iiwa_batter.trajectory_library import(
    Trajectory,
    LIBRARY_SPEEDS_MPH,
    LIBRARY_POSITIONS,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    make_trajectory_timesteps,
    perturb_vector,
    descent_step,
)

NUM_LOW_FIDELITY_ITERATIONS = 10
NUM_LOW_FIDELITY_WORKERS = 1


def measure_ball_pitch(pitch_speed_world, pitch_position_world, pitch_speed_measurement_error, pitch_position_measurement_error):
    """Measure the ball's position and speed in the world right after it's pitched.
    
    The way I've set this up the pitch position and speed determine where the ball is going to cross the plate, not where it is presently.
    """
    speed_noise = np.random.normal(0, pitch_speed_measurement_error)
    position_noise_y = np.random.normal(0, pitch_position_measurement_error)
    position_noise_z = np.random.normal(0, pitch_position_measurement_error)

    measured_pitch_speed = pitch_speed_world + speed_noise
    measured_pitch_position = [pitch_position_world[0], pitch_position_world[1] + position_noise_y, pitch_position_world[2] + position_noise_z]

    return measured_pitch_speed, measured_pitch_position

def measure_ball(ball_position_world, ball_velocity_world, ball_position_measurement_error, ball_velocity_measurement_error):
    """Measure the ball's position and speed in the world.
    
    This is the ball's position and velocity at the present moment.
    """
    position_noise = np.random.normal(0, ball_position_measurement_error, 3)
    velocity_noise = np.random.normal(0, ball_velocity_measurement_error, 3)

    measured_ball_position = ball_position_world + position_noise
    measured_ball_velocity = ball_velocity_world + velocity_noise

    return measured_ball_position, measured_ball_velocity

def measure_joints(joint_position_world, joint_velocity_world, joint_position_measurement_error, joint_velocity_measurement_error):
    """Measure the robot's joint positions and velocities in the world.
    
    This is the robot's joint positions and velocities at the present moment.
    """
    position_noise = np.random.normal(0, joint_position_measurement_error, NUM_JOINTS)
    velocity_noise = np.random.normal(0, joint_velocity_measurement_error, NUM_JOINTS)

    measured_joint_positions = joint_position_world + position_noise
    measured_joint_velocities = joint_velocity_world + velocity_noise

    return measured_joint_positions, measured_joint_velocities

def find_next_actions(
    robot,
    low_fidelity_simulators,
    low_fidelity_diagrams,
    original_trajectory,
    measured_joint_positions,
    measured_joint_velocities,
    joint_position_sample_distribution,
    joint_velocity_sample_distribution,
    measured_ball_position,
    measured_ball_velocity,
    ball_position_sample_distribution,
    ball_velocity_sample_distribution,
    start_time,
    ball_flight_time,
    learning_rate=2,
):
    """Find the next actions to take based on the measured present state.
    
    The starting point is the last control vector from this process.
    To account for the uncertainties in the low-fideltity simulation, we slightly vary the ball's position and velocity
    and run the same trajectory against these variations.
    Therefore, a good trajectory will be one that is robust to these variations.
    """
    torque_constraints = np.array([int(torque) for torque in JOINT_CONSTRAINTS[robot]["torque"].values()])
    present_control_vector = np.array([torques for torques in original_trajectory.values()])
    
    ball_positions = []
    ball_velocities = []
    for i in range(NUM_LOW_FIDELITY_WORKERS):
        ball_position_noise = np.random.normal(0, ball_position_sample_distribution)
        ball_velocity_noise = np.random.normal(0, ball_velocity_sample_distribution)
        ball_positions.append(measured_ball_position + ball_position_noise)
        ball_velocities.append(measured_ball_velocity + ball_velocity_noise)
    
    best_average_reward = -np.inf
    for i in range(NUM_LOW_FIDELITY_ITERATIONS):
        present_rewards = []
        present_trajectory = make_torque_trajectory(present_control_vector, ball_flight_time)
        for j in range(NUM_LOW_FIDELITY_WORKERS):
            reward = partial_trajectory_reward(
                simulator=low_fidelity_simulators[j],
                diagram=low_fidelity_diagrams[j],
                start_time=start_time,
                initial_joint_positions=measured_joint_positions,
                initial_joint_velocities=measured_joint_velocities,
                initial_ball_position=ball_positions[j],
                initial_ball_velocity=ball_velocities[j],
                torque_trajectory=present_trajectory,
            )
            present_rewards.append(reward)
        
        present_average_reward = np.mean(present_rewards)
        if present_average_reward > best_average_reward:
            best_average_reward = present_average_reward
            best_control_vector = present_control_vector

        if i > NUM_LOW_FIDELITY_ITERATIONS - 1:
            break

        perturbed_control_vector = perturb_vector(present_control_vector, learning_rate, torque_constraints, -torque_constraints)
        perturbed_rewards = []
        for j in range(NUM_LOW_FIDELITY_WORKERS):
            reward = partial_trajectory_reward(
                simulator=low_fidelity_simulators[j],
                diagram=low_fidelity_diagrams[j],
                start_time=start_time,
                initial_joint_positions=measured_joint_positions,
                initial_joint_velocities=measured_joint_velocities,
                initial_ball_position=ball_positions[j],
                initial_ball_velocity=ball_velocities[j],
                torque_trajectory=make_torque_trajectory(perturbed_control_vector, ball_flight_time),
            )
            perturbed_rewards.append(reward)
        perturbed_average_reward = np.mean(perturbed_rewards)
        updated_control_vector = descent_step(
            present_control_vector,
            perturbed_control_vector,
            present_average_reward,
            perturbed_average_reward,
            learning_rate,
            torque_constraints,
            -torque_constraints,
        )

        present_control_vector = updated_control_vector

    next_trajectory = make_torque_trajectory(best_control_vector, ball_flight_time)

    return next_trajectory


def real_time_operation(
    robot,
    pitch_speed_world,
    pitch_position_world,
    pitch_speed_measurement_error,
    pitch_position_measurement_error,
    joint_position_measurement_error=0,
    joint_velocity_measurement_error=0,
    joint_position_sample_distribution=0,
    joint_velocity_sample_distribution=0,
    ball_position_measurement_error=0,
    ball_velocity_measurement_error=0,
    ball_position_sample_distribution=0,
    ball_velocity_sample_distribution=0,
):
    """Real-time operation, in the sense that the CONTACT_DT is the world (taken as truth), and PITCH_DT is the low-fidelity simulation.
    
    What we're doing is simulating reality, but planning using a low-fidelity simulation.
    """

    # Set up the 'reality' case. This is our ground truth.
    simulator_world, diagram_world = setup_simulator(
        torque_trajectory={},
        model_urdf=robot,
        dt=CONTACT_DT,
        robot_constraints=JOINT_CONSTRAINTS[robot],
    )
    ball_initial_velocity_world, ball_time_of_flight_world = find_ball_initial_velocity(pitch_speed_world, pitch_position_world)
    control_timesteps_world = make_trajectory_timesteps(ball_time_of_flight_world)
    
    # Measure the ball's position and speed in the world at the time of the pitch (probably higher uncertainty here)
    measured_pitch_speed, measured_pitch_position = measure_ball_pitch(pitch_speed_world, pitch_position_world, pitch_speed_measurement_error, pitch_position_measurement_error)

    # Find the initial trajectory from our library which is closest to the measured pitch
    closest_speed = min(LIBRARY_SPEEDS_MPH, key=lambda x: abs(x - measured_pitch_speed))
    closest_position = min(LIBRARY_POSITIONS, key=lambda x: np.linalg.norm(np.array(x) - np.array(measured_pitch_position)))
    _, closest_time_of_flight = find_ball_initial_velocity(closest_speed, closest_position)

    # Make the robot start with the initial position but don't do anything yet
    start_trajectory_loader = Trajectory(robot, closest_speed, closest_position, "fine")
    start_initial_position, start_control_vector, _ = start_trajectory_loader.load_best_trajectory()
    run_swing_simulation(
        simulator=simulator_world,
        diagram=diagram_world,
        start_time=0,
        end_time=0,
        initial_joint_positions=start_initial_position,
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=ball_initial_velocity_world,
    )
    # Initial guess at planned trajectory is what we got from the library
    planned_trajectory = make_torque_trajectory(start_control_vector, closest_time_of_flight)

    # Set up the workers for low-fidelity simulation
    low_fidelity_simulators = []
    low_fidelity_diagrams = []
    for i in range(NUM_LOW_FIDELITY_WORKERS):
        simulator, diagram = setup_simulator(
            torque_trajectory={},
            model_urdf=robot,
            dt=PITCH_DT,
            robot_constraints=JOINT_CONSTRAINTS[robot],
        )
        low_fidelity_simulators.append(simulator)
        low_fidelity_diagrams.append(diagram)

    # Now is the fun part: in the interval between the control timesteps, plan using the low-fidelity simulation
    # Then, apply the trajectory to the world simulation. And repeat!
    # So while the world is going from 0 to dt,
    # search for the action to take at dt
    # Then while the world is going from dt to 2*dt,
    # search for the action to take at 2*dt
    taken_trajectory = {}
    for i, control_timestep in enumerate(control_timesteps_world):
        if i == len(control_timesteps_world) - 1:
            break
        # Get the present state of the world
        present_joint_positions, present_joint_velocities = parse_simulation_state(simulator_world, diagram_world, "iiwa")
        present_ball_position, present_ball_velocity = parse_simulation_state(simulator_world, diagram_world, "ball")
        # Measure the present state of the world
        measured_joint_positions, measured_joint_velocities = measure_joints(present_joint_positions, present_joint_velocities, joint_position_measurement_error, joint_velocity_measurement_error)
        measured_ball_position, measured_ball_velocity = measure_ball(present_ball_position, present_ball_velocity, ball_position_measurement_error, ball_velocity_measurement_error)

        # Take the action that was planned for this timestep
        taken_trajectory[control_timestep] = planned_trajectory[control_timestep]
        reset_systems(diagram_world, taken_trajectory)
        status_dict = run_swing_simulation(
            simulator_world,
            diagram_world,
            control_timestep,
            control_timesteps_world[i+1],
            present_joint_positions,
            present_joint_velocities,
            present_ball_position,
            present_ball_velocity,
        )
        if status_dict["result"] == "collision":
            break

        # While that's going on, plan the next action based on the measured present state
        planned_trajectory = find_next_actions(
            robot=robot,
            low_fidelity_simulators=low_fidelity_simulators,
            low_fidelity_diagrams=low_fidelity_diagrams,
            original_trajectory=planned_trajectory,
            measured_joint_positions=measured_joint_positions,
            measured_joint_velocities=measured_joint_velocities,
            joint_position_sample_distribution=joint_position_sample_distribution,
            joint_velocity_sample_distribution=joint_velocity_sample_distribution,
            measured_ball_position=measured_ball_position,
            measured_ball_velocity=measured_ball_velocity,
            ball_position_sample_distribution=ball_position_sample_distribution,
            ball_velocity_sample_distribution=ball_velocity_sample_distribution,
            start_time=control_timestep,
            ball_flight_time=ball_time_of_flight_world,
        )

    return taken_trajectory, status_dict


if __name__ == "__main__":
    np.random.seed(0)
    robot = "iiwa14"
    library_position_index = 0
    real_time_operation(
        robot="iiwa14",
        pitch_speed_world=LIBRARY_SPEEDS_MPH[0],
        pitch_position_world=LIBRARY_POSITIONS[library_position_index],
        pitch_speed_measurement_error=0.1,
        pitch_position_measurement_error=0.1,
    )