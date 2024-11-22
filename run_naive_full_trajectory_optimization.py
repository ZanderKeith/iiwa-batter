import numpy as np
import matplotlib.pyplot as plt

from iiwa_batter import PACKAGE_ROOT, CONTROL_DT, CONTACT_DT, PITCH_DT
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.physics import find_ball_initial_velocity
from iiwa_batter.save_load import trajectory_dir, save_trajectory

from iiwa_batter.swing_optimization.full_trajectory import setup_simulator, initialize_control_vector, stochastic_optimization_full_trajectory

trajectory_settings = {
    "robot": "iiwa14",
    "pitch_speed_mph": 90,
    "target_position": [0, 0, 0.6],
}
optimization_name = "naive_full_trajectory"

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
crash_iteration = 310 # The value of i when you do the traceback
save_interval = 1000
for i in range(10000):
    updated_control_vector, reward, reward_difference = stochastic_optimization_full_trajectory(simulator, station, robot_constraints, control_vector, control_timesteps, ball_initial_velocity, time_of_flight)
    rewards.append(reward)
    reward_differences.append(reward_difference)
    if reward > best_reward:
        best_control_vector = control_vector
        best_reward = reward
        print(f"Iteration {i}: {reward}")
    control_vector = updated_control_vector

    if i % save_interval == 0 or i == crash_iteration-1:
        save_trajectory(trajectory_settings, f"{optimization_name}_step_{i}", control_vector, reward, rewards, reward_differences)

print(f"Best reward: {best_reward}")
print(f"Difference variance: {np.var(reward_differences)}")

save_trajectory(trajectory_settings, optimization_name, best_control_vector, best_reward, rewards, reward_differences)

# Plot rewards over iteration
plt.plot(rewards)
plt.savefig(f"{trajectory_dir(trajectory_settings, optimization_name)}/reward_plot.png")