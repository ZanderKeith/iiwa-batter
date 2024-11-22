import numpy as np
from pydrake.all import (
    Diagram,
)

from iiwa_batter import NUM_JOINTS
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


def compare_dt_final_state():
    # See how the final state of the system changes with different values of dt
    pass
