import numpy as np
import dill
import time

from iiwa_batter import (
    PITCH_DT,
    CONTROL_DT,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import (
    setup_simulator,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    find_initial_positions,
    initialize_control_vector,
)
from iiwa_batter.swing_optimization.full_trajectory import (
    single_full_trajectory_torque_and_position,
)

def run_naive_full_trajectory_optimization(
    robot,
    target_velocity_mph,
    target_position,
    optimization_name,
    save_directory,
    simulation_dt=PITCH_DT,
    inner_iterations=10,
    outer_iterations=10,
    debug_prints=False,
):

    start_time = time.time()

    np.random.seed(0)
    robot_constraints = JOINT_CONSTRAINTS[robot]
    torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])
    position_constraints_upper = np.array([joint[1] for joint in robot_constraints["joint_range"].values()])
    position_constraints_lower = np.array([joint[0] for joint in robot_constraints["joint_range"].values()])

    simulator, diagram = setup_simulator(torque_trajectory={}, dt=simulation_dt, robot_constraints=robot_constraints)
    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(target_velocity_mph, target_position)
    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    #present_initial_position = find_initial_positions(simulator, diagram, robot_constraints, 1)[0]
    #present_control_vector = initialize_control_vector(robot_constraints, len(trajectory_timesteps))
    present_initial_position = np.ones(7)
    present_control_vector = np.ones((len(trajectory_timesteps), 7))

    training_results = {}
    best_reward = -np.inf
    for i in range(outer_iterations):
        next_initial_position, next_control_vector, present_reward = single_full_trajectory_torque_and_position(
            simulator=simulator,
            diagram=diagram,
            original_initial_joint_positions=present_initial_position,
            original_control_vector=present_control_vector,
            ball_initial_velocity=ball_initial_velocity,
            ball_time_of_flight=ball_time_of_flight,
            position_constraints_upper=position_constraints_upper,
            position_constraints_lower=position_constraints_lower,
            torque_constraints=torque_constraints,
            iterations=inner_iterations,
            return_best=False,
            learning_rate=1
        )

        if present_reward > best_reward:
            best_reward = present_reward
            best_initial_position = present_initial_position
            best_control_vector = present_control_vector

        training_results[i] = {
            "present_reward": present_reward,
            "present_initial_position": present_initial_position,
            "present_control_vector": present_control_vector,
            "best_reward_so_far": best_reward,
        }

        if i < outer_iterations - 1:
            present_initial_position = next_initial_position
            present_control_vector = next_control_vector

        if debug_prints:
            print(f"Top-Level Iteration {i}: present reward: {present_reward:.3f}, best reward: {best_reward:.3f}")

    training_results["total_time"] = time.time() - start_time
    training_results["best_initial_position"] = best_initial_position
    training_results["best_control_vector"] = best_control_vector
    training_results["final_best_reward"] = best_reward
    if debug_prints:
        print(f"Total time: {training_results['total_time']:.1f} seconds")

    with open(f"{save_directory}/{optimization_name}.dill", "wb") as f:
        dill.dump(training_results, f)
