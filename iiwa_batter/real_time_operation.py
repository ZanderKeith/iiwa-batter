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
from iiwa_batter.trajectory_library import(
    Trajectory,
    LIBRARY_SPEEDS_MPH,
    LIBRARY_POSITIONS,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
)

NUM_LOW_FIDELITY_ITERATIONS = 10
NUM_LOW_FIDELITY_WORKERS = 2


def measure_ball(pitch_speed_world, pitch_position_world, pitch_speed_measurement_error, pitch_position_measurement_error):
    """Measure the ball's position and speed in the world.
    
    The way I've set this up the 'position' is where the ball is going to cross the plate, not where it is presently.
    """
    speed_noise = np.random.normal(0, pitch_speed_measurement_error)
    position_noise_y = np.random.normal(0, pitch_position_measurement_error)
    position_noise_z = np.random.normal(0, pitch_position_measurement_error)

    measured_pitch_speed = pitch_speed_world + speed_noise
    measured_pitch_position = [pitch_position_world[0], pitch_position_world[1] + position_noise_y, pitch_position_world[2] + position_noise_z]

    return measured_pitch_speed, measured_pitch_position


def find_next_actions(
    pitch_speed_sample_distribution,
    pitch_position_sample_distribution,
):
    pass


def real_time_operation(
    robot,
    pitch_speed_world,
    pitch_position_world,
    pitch_speed_measurement_error,
    pitch_position_measurement_error,
    pitch_speed_sample_distribution,
    pitch_position_sample_distribution,
    meshcat=None,
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
    control_timesteps_world = np.arange(0, ball_time_of_flight_world+CONTROL_DT, CONTROL_DT)
    
    # Measure the ball's position and speed in the world
    measured_pitch_speed, measured_pitch_position = measure_ball(pitch_speed_world, pitch_position_world, pitch_speed_measurement_error, pitch_position_measurement_error)

    # Find the initial trajectory from our library which is closest to the measured pitch
    closest_speed = min(LIBRARY_SPEEDS_MPH, key=lambda x: abs(x - measured_pitch_speed))
    closest_position = min(LIBRARY_POSITIONS, key=lambda x: np.linalg.norm(np.array(x) - np.array(measured_pitch_position)))
    _, closest_time_of_flight = find_ball_initial_velocity(closest_speed, closest_position)

    # Start the trajectory using the path from this initial position
    start_trajectory_loader = Trajectory(robot, closest_speed, closest_position, "fine")
    start_initial_position, start_control_vector, _ = start_trajectory_loader.load_best_trajectory()
    start_trajectory_timesteps = np.arange(0, closest_time_of_flight+CONTROL_DT, CONTROL_DT)
    start_trajectory = make_torque_trajectory(start_control_vector, start_trajectory_timesteps)
    reset_systems(diagram_world, start_trajectory)
    # Make the robot start with the initial position but don't do anything yet
    run_swing_simulation(
        simulator=simulator_world,
        diagram=diagram_world,
        start_time=0,
        end_time=0,
        initial_joint_positions=start_initial_position,
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=ball_initial_velocity_world,
        setup_only=True,
    )

    taken_trajectory = {0: start_control_vector[0]}

    # Now is the fun part: in the interval between the control timesteps, plan using the low-fidelity simulation
    # Then, apply the trajectory to the world simulation. And repeat!
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

    # So while the world is going from 0 to dt,
    # search for the action to take at dt
    # Then while the world is going from dt to 2*dt,
    # search for the action to take at 2*dt

    for i, control_timestep in enumerate(control_timesteps_world):
        # Measure the present state of the world
        present_joint_positions, present_joint_velocities = parse_simulation_state(simulator_world, diagram_world, "iiwa")
        present_ball_position, present_ball_velocity = parse_simulation_state(simulator_world, diagram_world, "ball")

        #updated_trajectory = find_next_actions()
        updated_trajectory = {0: np.zeros(NUM_JOINTS)}
        taken_trajectory[control_timestep] = updated_trajectory[control_timestep]

        # Apply the control to the world simulation
        reset_systems(diagram_world, updated_trajectory)
        run_swing_simulation(
            simulator_world,
            diagram_world,
            control_timestep,
            control_timesteps_world[i+1],
            present_joint_positions,
            present_joint_velocities,
            present_ball_position,
            present_ball_velocity,
        )


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
        pitch_speed_sample_distribution=None,
        pitch_position_sample_distribution=None,
    )