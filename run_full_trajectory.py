import json
import numpy as np
from pydrake.geometry import StartMeshcat
import dill

from iiwa_batter import PACKAGE_ROOT, CONTROL_TIMESTEP
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.physics import find_initial_velocity, PITCH_START_POSITION

from iiwa_batter.swing_optimization.full_trajectory import run_full_trajectory

np.random.seed(0)
robot_constraints = JOINT_CONSTRAINTS["kr6r900"] 

initial_velocity, time_of_flight = find_initial_velocity(90, [0, 0, 0.6])

timesteps = np.arange(0, time_of_flight, CONTROL_TIMESTEP)

def initialize_control_vector(robot_constraints, num_timesteps):
    # First index is the initial position
    # All the next ones are the control torques
    num_joints = len(robot_constraints["torque"])
    control_vector = np.zeros(num_joints + num_timesteps*num_joints)

    # Set the initial position
    for i, joint in enumerate(robot_constraints["joint_range"].values()):
        control_vector[i] = np.random.uniform(joint[0], joint[1])

    for t in range(num_timesteps):
        for i, torque in enumerate(robot_constraints["torque"].values()):
            control_vector[num_joints + t*num_joints + i] = np.random.uniform(-torque, torque)

    return control_vector

def make_torque_trajectory(control_vector, robot_constraints, timesteps):
    torque_trajectory = {}
    for i, timestep in enumerate(timesteps):
        torque_trajectory[timestep] = control_vector[len(robot_constraints["joint_range"])*i:len(robot_constraints["joint_range"])*(i+1)]
    return torque_trajectory

def stochastic_optimization(original_control_vector, robot_constraints, timesteps, learning_rate):
    position_variance = 0.1 # About 1 degree
    torque_variance= 1 # About 1% of the max torque

    num_joints = len(robot_constraints["torque"])

    # Determine the loss from this control vector
    torque_trajectory = make_torque_trajectory(original_control_vector, robot_constraints, timesteps)
    original_reward = run_full_trajectory(None, original_control_vector[:num_joints], [PITCH_START_POSITION, initial_velocity], time_of_flight, robot_constraints, torque_trajectory)

    # Perturb the control vector, ensuring that the joint constraints are still satisfied
    perturbed_vector = np.empty_like(original_control_vector)
    for i, joint in enumerate(robot_constraints["joint_range"].values()):
        perturbation = np.random.normal(0, position_variance)
        capped_perturbation = np.clip(control_vector[i] + perturbation, joint[0], joint[1])
        perturbed_vector[i] = capped_perturbation - original_control_vector[i]

    for t in range(len(timesteps)):
        for i, torque in enumerate(robot_constraints["torque"].values()):
            perturbation = np.random.normal(0, torque_variance)
            capped_perturbation = np.clip(control_vector[num_joints + t*num_joints + i] + perturbation, -torque, torque)
            perturbed_vector[num_joints + t*num_joints + i] = capped_perturbation - original_control_vector[num_joints + t*num_joints + i]

    perturbed_control_vector = original_control_vector + perturbed_vector
    perturbed_torque_trajectory = make_torque_trajectory(perturbed_control_vector, robot_constraints, timesteps)

    perturbed_reward = run_full_trajectory(None, perturbed_control_vector[:num_joints], [PITCH_START_POSITION, initial_velocity], time_of_flight, robot_constraints, perturbed_torque_trajectory)

    updated_control_vector = original_control_vector + learning_rate * (perturbed_reward - original_reward) * perturbed_vector

    return updated_control_vector, original_reward, perturbed_reward - original_reward

control_vector = initialize_control_vector(robot_constraints, len(timesteps))
best_control_vector = control_vector
best_reward = -np.inf
rewards = []
reward_differences = []
for i in range(1000):
    control_vector, reward, reward_difference = stochastic_optimization(control_vector, robot_constraints, timesteps, 0.1)
    rewards.append(reward)
    reward_differences.append(reward_difference)
    if reward > best_reward:
        best_control_vector = control_vector
        best_reward = reward

print(best_reward)
print(f"Difference variance: {np.var(reward_differences)}")
import matplotlib.pyplot as plt

# Plot rewards over iteration
plt.plot(rewards)

plt.savefig("rewards.png")

# Save the best control vector, the rewards, and the reward differences
with open("best_control_vector.dill", "wb") as f:
    dill.dump((best_control_vector, rewards, reward_differences), f)