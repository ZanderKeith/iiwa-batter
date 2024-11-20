import os
import dill
import numpy as np
import matplotlib.pyplot as plt

from iiwa_batter import PACKAGE_ROOT, CONTROL_DT, CONTACT_DT, PITCH_DT
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.physics import find_ball_initial_velocity

from iiwa_batter.swing_optimization.full_trajectory import setup_simulator, initialize_control_vector, stochastic_optimization_full_trajectory

trajectory_settings = {
    "robot": "kr6r900",
    "pitch_speed_mph": 90,
    "target_position": [0, 0, 0.6],
}

robot = trajectory_settings["robot"]
pitch_speed_mph = trajectory_settings["pitch_speed_mph"]
target_position_y = trajectory_settings["target_position"][1]
target_position_z = trajectory_settings["target_position"][2]

robot_constraints = JOINT_CONSTRAINTS[robot] 

ball_initial_velocity, time_of_flight = find_ball_initial_velocity(trajectory_settings["pitch_speed_mph"], 
                                                         trajectory_settings["target_position"])

control_timesteps = np.arange(0, time_of_flight, CONTROL_DT)

simulator, station = setup_simulator(dt=PITCH_DT)

np.random.seed(0)
control_vector = initialize_control_vector(robot_constraints, len(control_timesteps))
best_control_vector = control_vector
best_reward = -np.inf
rewards = []
reward_differences = []
for i in range(10):
    control_vector, reward, reward_difference = stochastic_optimization_full_trajectory(simulator, station, robot_constraints, control_vector, control_timesteps, ball_initial_velocity, time_of_flight)
    rewards.append(reward)
    reward_differences.append(reward_difference)
    if reward > best_reward:
        best_control_vector = control_vector
        best_reward = reward
    if i % 10 == 0:
        print(f"Iteration {i}: {reward}")

print(best_reward)
print(f"Difference variance: {np.var(reward_differences)}")

# Save results
save_directory = f"{PACKAGE_ROOT}/swing_optimization/trajectories/naive_full_trajectory/{robot}_{pitch_speed_mph}mph_y{target_position_y}_z{target_position_z}"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Plot rewards over iteration
plt.plot(rewards)
plt.savefig(f"{save_directory}/reward_plot.png")

# Save the best control vector, the rewards, and the reward differences
with open(f"{save_directory}/best_trajectory.dill", "wb") as f:
   dill.dump((best_control_vector, best_reward, rewards, reward_differences), f)