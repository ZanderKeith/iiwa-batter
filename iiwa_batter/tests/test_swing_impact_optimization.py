import dill
import numpy as np

from iiwa_batter import (
    NUM_JOINTS,
    PACKAGE_ROOT,
    CONTACT_DT,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
)
from iiwa_batter.swing_simulator import (
    setup_simulator,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.partial_trajectory import partial_trajectory_reward
from iiwa_batter.swing_optimization.stochastic_gradient_descent import make_torque_trajectory, make_trajectory_timesteps
from iiwa_batter.trajectory_library import LIBRARY_SPEEDS_MPH, LIBRARY_POSITIONS

from iiwa_batter.swing_optimization.swing_impact import (
    dummy_torque_trajectory,
    calculate_plate_time_and_ball_state,
    run_swing_impact_optimization,
)

def test_run_swing_impact_optimization_single():
    # Ensure we can successfully restore the optimization results from a saved file

    np.random.seed(0)

    robot = "iiwa14"
    target_velocity_mph = LIBRARY_SPEEDS_MPH[0]
    target_position = LIBRARY_POSITIONS[0]
    optimization_name = "test_save_load_consistency_single_swing_impact"
    save_directory = f"{PACKAGE_ROOT}/tests/trajectories"
    test_dt = CONTACT_DT

    plate_time, plate_ball_postion, plate_ball_velocity = calculate_plate_time_and_ball_state(target_velocity_mph, target_position, test_dt)
    run_swing_impact_optimization(
        robot=robot,
        optimization_name=optimization_name,
        save_directory=save_directory,
        plate_time=plate_time,
        plate_ball_position=plate_ball_postion,
        plate_ball_velocity=plate_ball_velocity,
        present_joint_positions=np.zeros(NUM_JOINTS),
        present_joint_velocities=np.zeros(NUM_JOINTS),
        learning_rate=1,
        iterations=1,
        debug_prints=False,
    )

    with open(f"{save_directory}/{optimization_name}.dill", "rb") as f:
        results_dict = dill.load(f)

    initial_joint_positions = results_dict["best_joint_positions"]
    initial_joint_velocities = results_dict["best_joint_velocities"]
    simulator, diagram = setup_simulator(
        torque_trajectory = dummy_torque_trajectory(plate_time),
        model_urdf=robot,
        dt=test_dt,
        robot_constraints=JOINT_CONSTRAINTS[robot],
    )

    reward = partial_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        start_time=plate_time,
        initial_joint_positions=initial_joint_positions,
        initial_joint_velocities=initial_joint_velocities,
        initial_ball_position=plate_ball_postion,
        initial_ball_velocity=plate_ball_velocity,
        torque_trajectory=dummy_torque_trajectory(plate_time),
    )

    assert results_dict["final_best_reward"] == reward

def test_run_swing_impact_optimization_multi():
    # Ensure we can successfully restore the optimization results from a saved file

    np.random.seed(0)

    robot = "iiwa14"
    target_velocity_mph = LIBRARY_SPEEDS_MPH[0]
    target_position = LIBRARY_POSITIONS[0]
    optimization_name = "test_save_load_consistency_multi_swing_impact"
    save_directory = f"{PACKAGE_ROOT}/tests/trajectories"
    test_dt = CONTACT_DT

    plate_time, plate_ball_postion, plate_ball_velocity = calculate_plate_time_and_ball_state(target_velocity_mph, target_position, test_dt)
    run_swing_impact_optimization(
        robot=robot,
        optimization_name=optimization_name,
        save_directory=save_directory,
        plate_time=plate_time,
        plate_ball_position=plate_ball_postion,
        plate_ball_velocity=plate_ball_velocity,
        present_joint_positions=np.zeros(NUM_JOINTS),
        present_joint_velocities=np.zeros(NUM_JOINTS),
        learning_rate=1,
        iterations=3,
        debug_prints=False,
    )

    with open(f"{save_directory}/{optimization_name}.dill", "rb") as f:
        results_dict = dill.load(f)

    initial_joint_positions = results_dict["best_joint_positions"]
    initial_joint_velocities = results_dict["best_joint_velocities"]
    simulator, diagram = setup_simulator(
        torque_trajectory = dummy_torque_trajectory(plate_time),
        model_urdf=robot,
        dt=test_dt,
        robot_constraints=JOINT_CONSTRAINTS[robot],
    )

    reward = partial_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        start_time=plate_time,
        initial_joint_positions=initial_joint_positions,
        initial_joint_velocities=initial_joint_velocities,
        initial_ball_position=plate_ball_postion,
        initial_ball_velocity=plate_ball_velocity,
        torque_trajectory=dummy_torque_trajectory(plate_time),
    )

    assert results_dict["final_best_reward"] == reward