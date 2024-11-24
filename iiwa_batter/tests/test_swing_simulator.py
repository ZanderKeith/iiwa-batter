import numpy as np
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
    find_ball_initial_velocity,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
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


def test_rectilinear_torque_trajectory():
    # Ensure the torque trajectory correctly follows rectilinear interpolation
    pass


def test_torque_trajectory_change_on_reset():
    # Ensure the torque trajectory can be changed on reset, and that the new trajectory is used
    torque_trajectory_1 = {0: np.array([0] * NUM_JOINTS), 1: np.array([1] * NUM_JOINTS)}
    torque_trajectory_2 = {0: np.array([2] * NUM_JOINTS), 1: np.array([3] * NUM_JOINTS)}

    simulator_1, diagram_1 = setup_simulator(torque_trajectory=torque_trajectory_1, add_contact=False)
    simulator_2, diagram_2 = setup_simulator(torque_trajectory=torque_trajectory_2, add_contact=False)

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

    torque_trajectory = {
        0: np.array([50] * NUM_JOINTS),
        0.2: np.array([100] * NUM_JOINTS),
    }

    simulator_contact_dt, diagram_contact_dt = setup_simulator(
        torque_trajectory=torque_trajectory, dt=CONTACT_DT
    )

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

    simulator_pitch_dt, diagram_pitch_dt = setup_simulator(
        torque_trajectory=torque_trajectory, dt=PITCH_DT
    )

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


def test_self_collision_check():
    # Ensure the collision check system works for self-collisions

    simulator_contact_dt, diagram_contact_dt = setup_simulator(
        torque_trajectory={}, dt=PITCH_DT
    )

    status_dict = run_swing_simulation(
        simulator=simulator_contact_dt,
        diagram=diagram_contact_dt,
        start_time=0,
        end_time=1.3,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
    )

    assert status_dict["result"] == "collision"
    assert status_dict["severity"] > 0


def test_non_self_collision_check():
    # Ensure the simulator doesn't report a collision when the ball and the bat collide
    simulator_contact_dt, diagram_contact_dt = setup_simulator(
        torque_trajectory={}, dt=PITCH_DT
    )

    initial_ball_velocity, _ = find_ball_initial_velocity(90, [0, 0.9, 1.2])

    status_dict = run_swing_simulation(
        simulator=simulator_contact_dt,
        diagram=diagram_contact_dt,
        start_time=0,
        end_time=0.5,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=initial_ball_velocity,
    )

    assert status_dict["result"] != "collision"
    assert "severity" not in status_dict.keys()


def test_collision_doesnt_affect_final_state():
    # Ensure adding the collision geometry doesn't affect the final state of the system

    simulator_collision, diagram_collision = setup_simulator(
        torque_trajectory={}, dt=PITCH_DT, add_contact=True
    )
    simulator_no_collision, diagram_no_collision = setup_simulator(
        torque_trajectory={}, dt=PITCH_DT, add_contact=False
    )

    run_swing_simulation(
        simulator=simulator_collision,
        diagram=diagram_collision,
        start_time=0,
        end_time=0.5,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
    )

    run_swing_simulation(
        simulator=simulator_no_collision,
        diagram=diagram_no_collision,
        start_time=0,
        end_time=0.5,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
    )

    joint_positions_collision, joint_velocities_collision = parse_simulation_state(
        simulator_collision, diagram_collision, "iiwa"
    )
    joint_positions_no_collision, joint_velocities_no_collision = parse_simulation_state(
        simulator_no_collision, diagram_no_collision, "iiwa"
    )

    assert np.allclose(joint_positions_collision, joint_positions_no_collision)
    assert np.allclose(joint_velocities_collision, joint_velocities_no_collision)


def test_joint_limits():
    # Ensure joint position limits are enforced
    robot_constraints = JOINT_CONSTRAINTS["iiwa14"]

    simulator, diagram = setup_simulator(torque_trajectory={0:np.ones(NUM_JOINTS)*80}, dt=PITCH_DT, add_contact=False, robot_constraints=robot_constraints)

    status_dict = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=2.0,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
        record_state=True,
    )

    # Ensure the joint positions and velocities are within the limits at all times
    state_dict = status_dict["state"]
    for time, state in state_dict.items():
        joint_positions = state["iiwa"][0]
        joint_velocities = state["iiwa"][1]
        for joint, positions in robot_constraints["joint_range"].items():
            assert positions[0]*1.01 <= joint_positions[int(joint)-1] <= positions[1]*1.01, f"Joint {joint} has position {joint_positions[int(joint)-1]} at time {time}"
            assert -robot_constraints["joint_velocity"][joint] <= joint_velocities[int(joint)-1] <= robot_constraints["joint_velocity"][joint], f"Joint {joint} has velocity {joint_velocities[int(joint)-1]} at time {time}"

    simulator, diagram = setup_simulator(torque_trajectory={0:np.ones(NUM_JOINTS)*-80}, dt=PITCH_DT, add_contact=False, robot_constraints=robot_constraints)

    status_dict = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=2.0,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
        record_state=True,
    )

    # Ensure the joint positions and velocities are within the limits at all times
    state_dict = status_dict["state"]
    for time, state in state_dict.items():
        joint_positions = state["iiwa"][0]
        joint_velocities = state["iiwa"][1]
        for joint, positions in robot_constraints["joint_range"].items():
            assert positions[0]*1.01 <= joint_positions[int(joint)-1] <= positions[1]*1.01, f"Joint {joint} has position {joint_positions[int(joint)-1]} at time {time}"
            assert -robot_constraints["joint_velocity"][joint] <= joint_velocities[int(joint)-1] <= robot_constraints["joint_velocity"][joint], f"Joint {joint} has velocity {joint_velocities[int(joint)-1]} at time {time}"

    # Ensure the cumulative limit break is greater than 1
    enforce_joint_limit_system = diagram.GetSubsystemByName("enforce_joint_limit_system")
    enforce_joint_limit_system_context = enforce_joint_limit_system.GetMyContextFromRoot(simulator.get_context())
    cumulative_limit_break = enforce_joint_limit_system.GetOutputPort("cumulative_limit_break").Eval(enforce_joint_limit_system_context)
    assert cumulative_limit_break[0] > 1


def test_simulator_time_advance():
    # This one is just for me, what actually happens when the simulator advances time?

    simulator, diagram = setup_simulator(torque_trajectory={}, dt=PITCH_DT, add_contact=False)

    run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=1,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
    )

    time_1 = parse_simulation_state(simulator, diagram, "time")

    simulator.AdvanceTo(time_1)

    time_2 = parse_simulation_state(simulator, diagram, "time")

    assert time_1 == time_2


def test_simulation_state_preservation():
    # With a torque trajectory that changes over time, run two sets of simulations
    # The first one goes all the way through
    # The second one resets but loads the state where it left off
    # Ensure the final states are the same

    robot_constraints = JOINT_CONSTRAINTS["iiwa14"]

    torque_trajectory = {
        0: np.ones(NUM_JOINTS) * 40, 
        CONTROL_DT: np.ones(NUM_JOINTS) * -40,
        CONTROL_DT*2: np.ones(NUM_JOINTS) * 40,
        CONTROL_DT*3: np.ones(NUM_JOINTS) * -40
    }

    # torque_trajectory = {
    #     0: np.zeros(NUM_JOINTS),
    # }

    simulator, diagram = setup_simulator(torque_trajectory=torque_trajectory, dt=PITCH_DT*20, add_contact=False, robot_constraints=None)

    status_dict_a = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=CONTROL_DT*4,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
        record_state=True,
    )

    joint_positions_a, joint_velocities_a = parse_simulation_state(simulator, diagram, "iiwa")
    ball_position_a, ball_velocity_a = parse_simulation_state(simulator, diagram, "ball")
    end_time_a = parse_simulation_state(simulator, diagram, "time")

    reset_simulator(simulator, diagram)

    status_dict_i = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=CONTROL_DT*2,
        initial_joint_positions=np.array([0] * NUM_JOINTS),
        initial_joint_velocities=np.array([0] * NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=np.zeros(3),
        record_state=True,
    )

    joint_positions_i, joint_velocities_i = parse_simulation_state(simulator, diagram, "iiwa")
    ball_position_i, ball_velocity_i = parse_simulation_state(simulator, diagram, "ball")
    final_torque = parse_simulation_state(simulator, diagram, "iiwa_actuation")

    reset_simulator(simulator, diagram)

    status_dict_b = run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=CONTROL_DT*2,
        end_time=CONTROL_DT*4,
        initial_joint_positions=joint_positions_i,
        initial_joint_velocities=joint_velocities_i,
        initial_ball_position=ball_position_i,
        initial_ball_velocity=ball_velocity_i,
        record_state=True,
    )
    new_simulator = Simulator(diagram)
    new_simulator.AdvanceTo(CONTROL_DT*4)

    joint_positions_b, joint_velocities_b = parse_simulation_state(new_simulator, diagram, "iiwa")
    ball_position_b, ball_velocity_b = parse_simulation_state(new_simulator, diagram, "ball")
    end_time_b = parse_simulation_state(new_simulator, diagram, "time")

    state_a = status_dict_a["state"]
    state_i = status_dict_i["state"]
    #state_b = status_dict_b["state"]

    assert end_time_a == end_time_b
    assert np.allclose(ball_position_a, ball_position_b)
    assert np.allclose(ball_velocity_a, ball_velocity_b) # These are fine

    joint_positions_deg_a = np.rad2deg(joint_positions_a)
    joint_positions_deg_b = np.rad2deg(joint_positions_b)
    joint_velocities_deg_a = np.rad2deg(joint_velocities_a)
    joint_velocities_deg_b = np.rad2deg(joint_velocities_b)

    # Ok, so the joints are within 0.1 degrees of each other, and the velocities are within 1 degree/s of each other
    assert np.allclose(joint_positions_deg_a, joint_positions_deg_b, atol=1e-1)
    assert np.allclose(joint_velocities_deg_a, joint_velocities_deg_b, atol=1)

def test_benchmark_simulation_handoff():
    # See how much faster it is to run the simulation with a longer dt during the pitch and a shorter dt during the swing
    pass