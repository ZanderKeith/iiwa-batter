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
    test_dt = PITCH_DT

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

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(target_velocity_mph, target_position)
    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
    simulator, diagram = setup_simulator(torque_trajectory, dt=test_dt, robot_constraints=JOINT_CONSTRAINTS[robot])

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

def test_contact_consistency():
    target_velocity_mph = 90
    target_position = [0, 0, 0.6]
    test_dt = PITCH_DT

    initial_joint_positions = np.ones(NUM_JOINTS)
    

    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(target_velocity_mph, target_position)
    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    control_vector = np.ones((len(trajectory_timesteps), NUM_JOINTS))
    torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
    simulator, diagram = setup_simulator(torque_trajectory, dt=test_dt)

    reward_1 = full_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        initial_joint_positions=initial_joint_positions,
        control_vector=control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
    )

    reward_2 = full_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        initial_joint_positions=initial_joint_positions,
        control_vector=control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
    )

    assert np.isclose(reward_1, reward_2)

