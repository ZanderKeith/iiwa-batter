import dill
import matplotlib.pyplot as plt

from iiwa_batter import (
    PACKAGE_ROOT,
    NUM_JOINTS,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

BACKGROUND_COLOR = "#1F1F1F"
TEXT_COLOR = "white"


def plot_joint_velocity(state_dict, robot_constraints, save_dir):
    max_velocities = [robot_constraints["joint_velocity"][str(joint + 1)] for joint in range(NUM_JOINTS)]

    trajectory_times = [time for time in state_dict.keys()]
    iiwa_joint_velocities = [state_dict[time]["iiwa"][1] for time in trajectory_times]
    
    velocity_percentage = [[(velocity/max_velocity)*100 for velocity, max_velocity in zip(joint_velocity, max_velocities)] for joint_velocity in iiwa_joint_velocities]

    fig, ax = plt.subplots()
    ax.plot(trajectory_times, velocity_percentage, label=[f"Joint {joint+1}" for joint in range(NUM_JOINTS)])
    ax.set_xlim(0, trajectory_times[-1])
    ax.set_ylim(-100, 100)

    ax.set_title("Joint Velocity", fontsize=17, color=TEXT_COLOR)
    ax.set_xlabel("Time [s]", fontsize=16, color=TEXT_COLOR)
    ax.set_ylabel("Velocity [%]", fontsize=16, color=TEXT_COLOR)
    ax.tick_params(axis='both', which='major', labelsize=14, colors=TEXT_COLOR)
    ax.legend(fontsize=14, loc="upper left")

    fig.patch.set_facecolor("#1F1F1F")

    fig.set_figwidth(8)

    fig.tight_layout()

    fig.savefig(save_dir, facecolor=BACKGROUND_COLOR)

# trajectory = Trajectory("iiwa14", 90, [0, 0, 0.6], "coarse_0")
# state_dict = trajectory.recover_trajectory_paths()

# robot_constraints = JOINT_CONSTRAINTS[trajectory.robot]
# plot_joint_velocity(state_dict, robot_constraints, f"{trajectory.data_directory()}/joint_velocity.png")

def plot_learning(results_dict, save_dir):

    present_rewards = [results_dict["learning"][time]["present_reward"] for time in results_dict["learning"].keys()]
    best_rewards = [results_dict["learning"][time]["best_reward_so_far"] for time in results_dict["learning"].keys()]

    fig, ax = plt.subplots()
    ax.plot(results_dict["learning"].keys(), present_rewards, label="Present Reward", color="green")
    ax.plot(results_dict["learning"].keys(), best_rewards, label="Best Reward So Far", color="red")

    ax.set_title(f"Learning Progress. Best reward: {results_dict['final_best_reward']:.2f}", fontsize=17, color=TEXT_COLOR)
    ax.set_xlabel("Iteration", fontsize=16, color=TEXT_COLOR)
    ax.set_ylabel("Reward", fontsize=16, color=TEXT_COLOR)
    ax.tick_params(axis='both', which='major', labelsize=14, colors=TEXT_COLOR)

    ax.legend(fontsize=14, loc="upper left")

    fig.savefig(save_dir, facecolor=BACKGROUND_COLOR)

robot = "iiwa14"
target_velocity_mph = 90
target_position = [0, 0, 0.6]
optimization_name = "impact_3"
save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_velocity_mph}_{target_position}"

with open(f"{save_directory}/{optimization_name}.dill", "rb") as f:
    results_dict = dill.load(f)

plot_learning(results_dict, f"{save_directory}/learning_{optimization_name}.png")