import numpy as np
from pydrake.all import (
    Diagram,
    Simulator,
)

from iiwa_batter import (
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    REALTIME_DT,
    NUM_JOINTS,
    PITCH_DT,
)
from iiwa_batter.physics import (
    PITCH_START_POSITION,
    find_ball_initial_velocity,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    reset_systems,
    run_swing_simulation,
    setup_simulator,
)
from iiwa_batter.trajectory_library import (
    Trajectory,
    LIBRARY_SPEEDS_MPH,
    LIBRARY_POSITIONS,
    MAIN_SPEED,
    MAIN_POSITION,
)
from iiwa_batter.real_time_operation import real_time_operation, BASE_TRAJECTORY

def test_compare_low_fidelity_dt_final_state():
    # See how the final state of the system changes with different values of dt
    # CONCLUSION: Final state is slightly different, but not by much.

    initial_ball_velocity, _ = find_ball_initial_velocity(90, [0, 0, 0.6])

    torque_trajectory = {
        0: np.array([50] * NUM_JOINTS),
        0.2: np.array([100] * NUM_JOINTS),
    }

    simulator_contact_dt, diagram_contact_dt = setup_simulator(
        torque_trajectory=torque_trajectory, model_urdf="iiwa14", dt=CONTACT_DT, add_contact=False
    )

    run_swing_simulation(
        simulator=simulator_contact_dt,
        diagram=diagram_contact_dt,
        start_time=0,
        end_time=0.6,
        initial_joint_positions=np.zeros(NUM_JOINTS),
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=initial_ball_velocity,
    )

    joint_positions_a, joint_velocities_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "iiwa"
    )

    simulator_pitch_dt, diagram_pitch_dt = setup_simulator(
        torque_trajectory=torque_trajectory, model_urdf="iiwa14", dt=REALTIME_DT, add_contact=False
    )

    run_swing_simulation(
        simulator=simulator_pitch_dt,
        diagram=diagram_pitch_dt,
        start_time=0,
        end_time=0.6,
        initial_joint_positions=np.zeros(NUM_JOINTS),
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=initial_ball_velocity,
    )

    joint_positions_b, joint_velocities_b = parse_simulation_state(
        simulator_pitch_dt, diagram_pitch_dt, "iiwa"
    )

    joint_positions_deg_a = np.rad2deg(joint_positions_a)
    joint_positions_deg_b = np.rad2deg(joint_positions_b)
    joint_velocities_deg_a = np.rad2deg(joint_velocities_a)
    joint_velocities_deg_b = np.rad2deg(joint_velocities_b)

    # Ok, so the joints are within 2 degrees of each other, and the velocities are within 10 degree/s of each other
    assert np.allclose(joint_positions_deg_a, joint_positions_deg_b, atol=2)
    assert np.allclose(joint_velocities_deg_a, joint_velocities_deg_b, atol=10)

    sweet_spot_position_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "sweet_spot"
    )
    sweet_spot_position_b = parse_simulation_state(
        simulator_pitch_dt, diagram_pitch_dt, "sweet_spot"
    )

    # The position of the sweet spot is also within 1.5 cm, which is good
    assert np.allclose(sweet_spot_position_a, sweet_spot_position_b, atol=0.015)

    ball_position_a, ball_velocity_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "ball"
    )
    ball_position_b, ball_velocity_b = parse_simulation_state(
        simulator_pitch_dt, diagram_pitch_dt, "ball"
    )

    # Ball is within 1cm (mostly in the x direction) and velocity is within 1 mm/s
    assert np.allclose(ball_position_a, ball_position_b, atol=1e-2)
    assert np.allclose(ball_velocity_a, ball_velocity_b, atol=1e-3)

def test_unmodified_trajectories_match():
    # Run real-time operation for a pitch and ensure the state after taking the trajectory without changing it is the same
    robot = "iiwa14"
    pitch_speed_world = LIBRARY_SPEEDS_MPH[0]
    pitch_position_world = LIBRARY_POSITIONS[0]

    taken_trajectory, real_time_status_dict, real_time_simulator, real_time_diagram = real_time_operation(
        robot=robot,
        pitch_speed_world=pitch_speed_world,
        pitch_position_world=pitch_position_world,
        pitch_speed_measurement_error=0,
        pitch_position_measurement_error=0,
        debug_mode=True,
    )

    end_time = parse_simulation_state(real_time_simulator, real_time_diagram, "time")

    # Now run the swing simulation with the taken trajectory
    # Find the initial trajectory from our library which is closest to the measured pitch
    closest_speed = min(LIBRARY_SPEEDS_MPH, key=lambda x: abs(x - pitch_speed_world))
    closest_position = min(LIBRARY_POSITIONS, key=lambda x: np.linalg.norm(np.array(x) - np.array(pitch_position_world)))
    _, closest_time_of_flight = find_ball_initial_velocity(closest_speed, closest_position)

    # Make the robot start with the initial position but don't do anything yet
    start_trajectory_loader = Trajectory(robot, closest_speed, closest_position, BASE_TRAJECTORY)
    start_initial_position, start_control_vector, _ = start_trajectory_loader.load_best_trajectory()
    initial_ball_velocity_world, flight_time = find_ball_initial_velocity(pitch_speed_world, pitch_position_world)
    simulator, diagram = setup_simulator(torque_trajectory=taken_trajectory, model_urdf=robot, dt=CONTACT_DT, robot_constraints=JOINT_CONSTRAINTS[robot])

    comparison_status_dict = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=end_time,
        initial_joint_positions=start_initial_position,
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=initial_ball_velocity_world,
        record_state=True,
    )

    # Ensure the final state of the robot is the same
    joint_positions_a, joint_velocities_a = parse_simulation_state(real_time_simulator, real_time_diagram, "iiwa")
    joint_positions_b, joint_velocities_b = parse_simulation_state(simulator, diagram, "iiwa")
    assert np.allclose(joint_positions_a, joint_positions_b)
    assert np.allclose(joint_velocities_a, joint_velocities_b)

    # Ensure the final state of the ball is the same
    ball_position_a, ball_velocity_a = parse_simulation_state(real_time_simulator, real_time_diagram, "ball")
    ball_position_b, ball_velocity_b = parse_simulation_state(simulator, diagram, "ball")
    assert np.allclose(ball_position_a, ball_position_b)
    assert np.allclose(ball_velocity_a, ball_velocity_b)

    # Ensure the final position of the sweet spot is the same
    sweet_spot_position_a = parse_simulation_state(real_time_simulator, real_time_diagram, "sweet_spot")
    sweet_spot_position_b = parse_simulation_state(simulator, diagram, "sweet_spot")
    assert np.allclose(sweet_spot_position_a, sweet_spot_position_b)

def test_uniform_initial_positions():
    # For every pitch in the library, make sure the initial joint positions are close to the same
    main_trajectory_loader = Trajectory("iiwa14", MAIN_SPEED, MAIN_POSITION, "main")
    main_initial_position, _, _ = main_trajectory_loader.load_best_trajectory()

    for speed in LIBRARY_SPEEDS_MPH:
        for position in LIBRARY_POSITIONS:
            trajectory_loader = Trajectory("iiwa14", speed, position, BASE_TRAJECTORY)
            initial_position, _, _ = trajectory_loader.load_best_trajectory()
            assert np.allclose(main_initial_position, initial_position, atol=1e-3) # Less than 0.1 degrees of difference