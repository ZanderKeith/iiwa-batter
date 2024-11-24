import numpy as np
import dill
import time

from iiwa_batter import (
    PACKAGE_ROOT,
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    meters_to_feet,
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

optimization_name = "simple_test_pitch_dt"
start_time = time.time()

np.random.seed(0)
robot_constraints = JOINT_CONSTRAINTS["iiwa14"]
torque_constraints = np.array([int(torque) for torque in robot_constraints["torque"].values()])
position_constraints_upper = np.array([joint[1] for joint in robot_constraints["joint_range"].values()])
position_constraints_lower = np.array([joint[0] for joint in robot_constraints["joint_range"].values()])

simulator, diagram = setup_simulator(torque_trajectory={}, dt=PITCH_DT, robot_constraints=robot_constraints, add_contact=True)
ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(90, [0, 0, 0.6])
trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
present_initial_position = find_initial_positions(simulator, diagram, robot_constraints, 1)[0]
present_control_vector = initialize_control_vector(robot_constraints, len(trajectory_timesteps))

training_results = {}
best_reward = -np.inf
best_initial_position = present_initial_position
best_control_vector = present_control_vector
for i in range(100):
    present_initial_position, present_control_vector, present_reward, next_initial_position, next_control_vector = single_full_trajectory_torque_and_position(
        simulator=simulator,
        diagram=diagram,
        original_initial_joint_positions=present_initial_position,
        original_control_vector=present_control_vector,
        ball_initial_velocity=ball_initial_velocity,
        ball_time_of_flight=ball_time_of_flight,
        position_constraints_upper=position_constraints_upper,
        position_constraints_lower=position_constraints_lower,
        torque_constraints=torque_constraints,
        iterations=10,
        return_best=False,
        learning_rate=1
    )

    if present_reward > best_reward:
        best_reward = present_reward
        best_initial_position = present_initial_position
        best_control_vector = present_control_vector

    training_results[i] = {
        "present_reward": present_reward,
        "present_initial_position": present_initial_position.copy(),
        "present_control_vector": present_control_vector.copy(),
        "best_reward": best_reward,
    }
    print(f"Top-Level Iteration {i}: present reward: {present_reward:.1f}, best reward: {best_reward:.1f}")

    present_initial_position = next_initial_position
    present_control_vector = next_control_vector


training_results["total_time"] = time.time() - start_time
training_results["best_initial_position"] = best_initial_position
training_results["best_control_vector"] = best_control_vector
print(f"Total time: {training_results['total_time']:.1f} seconds")

with open(f"{PACKAGE_ROOT}/swing_optimization/trajectories/naive_full_trajectory/{optimization_name}.dill", "wb") as f:
    dill.dump(training_results, f)
