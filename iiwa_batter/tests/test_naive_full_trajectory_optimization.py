import dill
import numpy as np

from iiwa_batter import (
    NUM_JOINTS,
    PACKAGE_ROOT,
    PITCH_DT,
    CONTROL_DT,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    PITCH_START_POSITION,
    FLIGHT_TIME_MULTIPLE,
)
from iiwa_batter.swing_simulator import (
    setup_simulator,
    run_swing_simulation,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.full_trajectory import full_trajectory_reward
from iiwa_batter.swing_optimization.stochastic_gradient_descent import make_torque_trajectory

from iiwa_batter.naive_full_trajectory_optimization import run_naive_full_trajectory_optimization

def test_save_load_consistency():
    # Ensure that we can successfully restore the optimization results from a saved file

    robot = "iiwa14"
    target_velocity_mph = 90
    target_position = [0, 0, 0.6]
    optimization_name = "test_save_load_consistency"
    save_directory = f"{PACKAGE_ROOT}/tests/trajectories"
    test_dt = PITCH_DT*10

    run_naive_full_trajectory_optimization(
        robot=robot,
        target_velocity_mph=target_velocity_mph,
        target_position=target_position,
        optimization_name=optimization_name,
        save_directory=save_directory,
        simulation_dt=test_dt,
        inner_iterations=1,
        outer_iterations=1,
        debug_prints=False,
    )

    with open(f"{save_directory}/{optimization_name}.dill", "rb") as f:
        results_dict = dill.load(f)

    initial_joint_positions = results_dict["best_initial_position"]
    control_vector = results_dict["best_control_vector"]
    robot_constraints = JOINT_CONSTRAINTS[robot]
    torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])
    position_constraints_upper = np.array([joint[1] for joint in robot_constraints["joint_range"].values()])
    position_constraints_lower = np.array([joint[0] for joint in robot_constraints["joint_range"].values()])

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(target_velocity_mph, target_position)
    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
    simulator, diagram = setup_simulator(torque_trajectory, dt=test_dt)

    # status_dict = run_swing_simulation(
    #     simulator=simulator,
    #     diagram=diagram,
    #     start_time=0,
    #     end_time=ball_time_of_flight*FLIGHT_TIME_MULTIPLE,
    #     initial_joint_positions=initial_joint_positions,
    #     initial_joint_velocities=np.zeros(NUM_JOINTS),
    #     initial_ball_position=PITCH_START_POSITION,
    #     initial_ball_velocity=ball_initial_velocity,
    # )

    reward = full_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        initial_joint_positions=initial_joint_positions,
        control_vector=control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
    )

    optimized_best_reward = results_dict["final_best_reward"]

    assert np.isclose(reward, optimized_best_reward)