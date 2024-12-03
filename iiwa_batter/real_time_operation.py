from multiprocessing import Pool

import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    REALTIME_DT,
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
    MAIN_SPEED,
    MAIN_POSITION,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    make_trajectory_timesteps,
    perturb_vector,
    descent_step,
    expand_torque_trajectory,
)

NUM_LOW_FIDELITY_ITERATIONS = 2
NUM_LOW_FIDELITY_WORKERS = 10
NUM_LOW_FIDELITY_TRAJECTORIES = NUM_LOW_FIDELITY_WORKERS * 2

# Can't pickle a simulator / diagram so we have to make these global to pass them to the workers
WORKER_SIMULATORS = []
WORKER_DIAGRAMS = []

LOW_FIDELITY_LEARNING_RATE = 10

BASE_TRAJECTORY = "tune_fine"


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


def worker_task(start_time, measured_joint_positions, measured_joint_velocities, ball_position, ball_velocity, worker_trajectory, worker_index):
    global WORKER_SIMULATORS
    global WORKER_DIAGRAMS
    reward = partial_trajectory_reward(
        simulator=WORKER_SIMULATORS[worker_index],
        diagram=WORKER_DIAGRAMS[worker_index],
        start_time=start_time, 
        initial_joint_positions=measured_joint_positions,
        initial_joint_velocities=measured_joint_velocities,
        initial_ball_position=ball_position, 
        initial_ball_velocity=ball_velocity, 
        torque_trajectory=worker_trajectory)
    return reward

def find_next_actions(
    robot,
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
    worker_pool,
    learning_rate=LOW_FIDELITY_LEARNING_RATE,
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
    for i in range(NUM_LOW_FIDELITY_TRAJECTORIES):
        ball_position_noise = np.random.normal(0, ball_position_sample_distribution)
        ball_velocity_noise = np.random.normal(0, ball_velocity_sample_distribution)
        ball_positions.append(measured_ball_position + ball_position_noise)
        ball_velocities.append(measured_ball_velocity + ball_velocity_noise)
    
    initial_average_reward = None
    for i in range(NUM_LOW_FIDELITY_ITERATIONS):
        present_rewards = []
        present_trajectory = make_torque_trajectory(present_control_vector, ball_flight_time)
        results = worker_pool.starmap(worker_task, [(start_time, measured_joint_positions, measured_joint_velocities, ball_positions[j], ball_velocities[j], present_trajectory, j) for j in range(NUM_LOW_FIDELITY_TRAJECTORIES)])
        for result in results:
            present_rewards.append(result)
        present_average_reward = np.mean(present_rewards)

        if initial_average_reward is None:
            initial_average_reward = present_average_reward
            best_average_reward = present_average_reward
            best_control_vector = present_control_vector
        if present_average_reward > best_average_reward:
            best_average_reward = present_average_reward
            best_control_vector = present_control_vector

        if i > NUM_LOW_FIDELITY_ITERATIONS - 1:
            break
        
        worker_trajectories = []
        for j in range(NUM_LOW_FIDELITY_TRAJECTORIES):
            perturbed_control_vector = perturb_vector(present_control_vector, learning_rate, torque_constraints, -torque_constraints)
            worker_trajectories.append(make_torque_trajectory(perturbed_control_vector, ball_flight_time))
        perturbed_rewards = []
        results = worker_pool.starmap(worker_task, [(start_time, measured_joint_positions, measured_joint_velocities, ball_positions[j], ball_velocities[j], worker_trajectories[j], j) for j in range(NUM_LOW_FIDELITY_TRAJECTORIES)])
        for result in results:
            perturbed_rewards.append(result)
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

    if best_average_reward < initial_average_reward:
        raise ValueError("The best average reward is less than the initial average reward. This is a problem!")
    else:
        print(f"Timestep {start_time:.2f} improvement: {best_average_reward - initial_average_reward:.4f}")
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
    debug_mode=False,
):
    """Real-time operation, in the sense that the CONTACT_DT is the world (taken as truth), and REALTIME_DT is the low-fidelity simulation.
    
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
    start_position_loader = Trajectory(robot, MAIN_SPEED, MAIN_POSITION, "main")
    start_initial_position, _, _ = start_position_loader.load_best_trajectory()
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
    start_trajectory_loader = Trajectory(robot, closest_speed, closest_position, BASE_TRAJECTORY)
    _, start_control_vector, _ = start_trajectory_loader.load_best_trajectory()
    planned_trajectory = make_torque_trajectory(start_control_vector, closest_time_of_flight)
    # Expand the planned trajectory to the full flight time
    planned_trajectory = expand_torque_trajectory(planned_trajectory, ball_time_of_flight_world+3*CONTROL_DT)

    # Set up the workers for low-fidelity simulation
    # I REALLY hate that we need to do this, but we can't pickle the simulator / diagram
    global WORKER_SIMULATORS
    WORKER_SIMULATORS = []
    global WORKER_DIAGRAMS
    WORKER_DIAGRAMS = []
    for i in range(NUM_LOW_FIDELITY_TRAJECTORIES):
        simulator, diagram = setup_simulator(
            torque_trajectory={},
            model_urdf=robot,
            dt=REALTIME_DT,
            robot_constraints=JOINT_CONSTRAINTS[robot],
        )
        WORKER_SIMULATORS.append(simulator)
        WORKER_DIAGRAMS.append(diagram)
    worker_pool = Pool(NUM_LOW_FIDELITY_WORKERS)

    # Now is the fun part: in the interval between the control timesteps, plan using the low-fidelity simulation
    # Then, apply the trajectory to the world simulation. And repeat!
    # So while the world is going from 0 to dt,
    # search for the action to take at dt
    # Then while the world is going from dt to 2*dt,
    # search for the action to take at 2*dt
    taken_trajectory = {}
    status_dict = None
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
        reset_systems(diagram_world, planned_trajectory)
        new_status_dict = run_swing_simulation(
            simulator=simulator_world,
            diagram=diagram_world,
            start_time=control_timestep,
            end_time=control_timesteps_world[i+1],
            initial_joint_positions=present_joint_positions,
            initial_joint_velocities=present_joint_velocities,
            initial_ball_position=present_ball_position,
            initial_ball_velocity=present_ball_velocity,
        )
        if new_status_dict["result"] == "collision":
            break
        # Remember that the 'status_dict' is updated for each control timestep
        if status_dict is None:
            status_dict = new_status_dict
        else:
            if status_dict["closest_approach"] > new_status_dict["closest_approach"]:
                status_dict["closest_approach"] = new_status_dict["closest_approach"]
            if status_dict["result"] == "hit" and new_status_dict["result"] == "miss":
                status_dict["result"] = "miss"

        # While that's going on, plan the next action based on the measured present state
        if not debug_mode:
            planned_trajectory = find_next_actions(
                robot=robot,
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
                worker_pool=worker_pool,
            )

    if not debug_mode:
        return taken_trajectory, status_dict
    else:
        return taken_trajectory, status_dict, simulator_world, diagram_world

if __name__ == "__main__":
    np.random.seed(0)
    robot = "iiwa14"
    library_position_index = 3
    taken_trajectory, status_dict = real_time_operation(
        robot="iiwa14",
        pitch_speed_world=LIBRARY_SPEEDS_MPH[0],
        pitch_position_world=LIBRARY_POSITIONS[library_position_index],
        pitch_speed_measurement_error=0,
        pitch_position_measurement_error=0,
        debug_mode=False
    )

    print(status_dict)