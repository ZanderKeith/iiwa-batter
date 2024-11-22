import numpy as np
from pydrake.all import (
    Diagram,
)

from iiwa_batter import (
    PACKAGE_ROOT,
    NUM_JOINTS,
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
)

from iiwa_batter.physics import (
    PITCH_START_POSITION,
    find_ball_initial_velocity,
)

from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    reset_simulator,
    run_swing_simulation,
    setup_simulator,
)


def test_setup_simulator():
    # Ensure the simulator and diagram are created with the correct subsystems

    simulator, diagram = setup_simulator(torque_trajectory={}, plot_diagram=True)
    assert simulator is not None
    assert diagram is not None

    # Try to get the hardware station from the diagram
    station: Diagram = diagram.GetSubsystemByName("station")
    assert station is not None

    # We can't get the plant directly from the diagram, but we can get it from the station
    plant: Diagram = station.GetSubsystemByName("plant")
    assert plant is not None

    # Access the torque trajectory system from the diagram
    torque_trajectory_system = diagram.GetSubsystemByName("torque_trajectory_system")
    assert torque_trajectory_system is not None


def test_torque_trajectory_change_on_reset():
    # Ensure the torque trajectory can be changed on reset, and that the new trajectory is used
    torque_trajectory_1 = {0: np.array([0] * NUM_JOINTS), 1: np.array([1] * NUM_JOINTS)}
    torque_trajectory_2 = {0: np.array([2] * NUM_JOINTS), 1: np.array([3] * NUM_JOINTS)}

    simulator_1, diagram_1 = setup_simulator(torque_trajectory=torque_trajectory_1)
    simulator_2, diagram_2 = setup_simulator(torque_trajectory=torque_trajectory_2)

    # Run the simulation with the first torque trajectory, reset, and run with the second torque trajectory
    run_swing_simulation(
        simulator=simulator_1,
        diagram=diagram_1,
        start_time=0,
        end_time=1,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=np.array([0, 0, 0]),
        initial_ball_velocity=np.array([0, 0, 0]),
    )

    joint_positions_a, joint_velocities_a = parse_simulation_state(
        simulator_1, diagram_1, "iiwa"
    )

    reset_simulator(simulator_1, diagram_1, new_torque_trajectory=torque_trajectory_2)

    run_swing_simulation(
        simulator=simulator_1,
        diagram=diagram_1,
        start_time=0,
        end_time=1,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=np.array([0, 0, 0]),
        initial_ball_velocity=np.array([0, 0, 0]),
    )

    joint_positions_b, joint_velocities_b = parse_simulation_state(
        simulator_1, diagram_1, "iiwa"
    )

    assert not np.allclose(joint_positions_a, joint_positions_b)
    assert not np.allclose(joint_velocities_a, joint_velocities_b)

    run_swing_simulation(
        simulator=simulator_2,
        diagram=diagram_2,
        start_time=0,
        end_time=1,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=np.array([0, 0, 0]),
        initial_ball_velocity=np.array([0, 0, 0]),
    )

    joint_positions_c, joint_velocities_c = parse_simulation_state(
        simulator_2, diagram_2, "iiwa"
    )

    assert np.allclose(joint_positions_b, joint_positions_c)
    assert np.allclose(joint_velocities_b, joint_velocities_c)


def test_compare_simulation_dt_final_state():
    # See how the final state of the system changes with different values of dt
    # CONCLUSION: Final state is slightly different, but not by much.

    initial_ball_velocity, _ = find_ball_initial_velocity(90, [0, 0, 0.6])

    torque_trajectory = {0: np.array([50] * NUM_JOINTS), 0.2: np.array([100] * NUM_JOINTS)}
    
    simulator_contact_dt, diagram_contact_dt = setup_simulator(torque_trajectory=torque_trajectory, dt=CONTACT_DT)

    run_swing_simulation(
        simulator=simulator_contact_dt,
        diagram=diagram_contact_dt,
        start_time=0,
        end_time=0.45,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=initial_ball_velocity,
    )

    joint_positions_a, joint_velocities_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "iiwa"
    )

    simulator_pitch_dt, diagram_pitch_dt = setup_simulator(torque_trajectory=torque_trajectory, dt=PITCH_DT)

    run_swing_simulation(
        simulator=simulator_pitch_dt,
        diagram=diagram_pitch_dt,
        start_time=0,
        end_time=0.45,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
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

    # Ok, so the joints are within 0.1 degrees of each other, and the velocities are within 1 degree/s of each other
    assert np.allclose(joint_positions_deg_a, joint_positions_deg_b, atol=1e-1)
    assert np.allclose(joint_velocities_deg_a, joint_velocities_deg_b, atol=1)

    sweet_spot_position_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "sweet_spot"
    )
    sweet_spot_position_b = parse_simulation_state(
        simulator_pitch_dt, diagram_pitch_dt, "sweet_spot"
    )

    # The position of the sweet spot is also within 1 mm, which is good
    assert np.allclose(sweet_spot_position_a, sweet_spot_position_b, atol=1e-3)

    ball_position_a, ball_velocity_a = parse_simulation_state(
        simulator_contact_dt, diagram_contact_dt, "ball"
    )
    ball_position_b, ball_velocity_b = parse_simulation_state(
        simulator_pitch_dt, diagram_pitch_dt, "ball"
    )

    # Ball is within 1cm (mostly in the x direction) and velocity is within 1 mm/s
    assert np.allclose(ball_position_a, ball_position_b, atol=1e-2)
    assert np.allclose(ball_velocity_a, ball_velocity_b, atol=1e-3)

