import dill
import numpy as np
import matplotlib.pyplot as plt

from iiwa_batter import (
    PACKAGE_ROOT,
    NUM_JOINTS,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

BACKGROUND_COLOR = "#000000"
TEXT_COLOR = "white"

TABLE_COLUMN_FONT_SIZE = 28


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

    fig.patch.set_facecolor(BACKGROUND_COLOR)

    fig.set_figwidth(8)

    fig.tight_layout()

    fig.savefig(save_dir, facecolor=BACKGROUND_COLOR)
    plt.close(fig)


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
    plt.close(fig)


def compare_joint_constraints(robot):
    # Make a table comparing the joint constraints
    IMPROVEMENT_COLOR = "#00FF00"
    WORSE_COLOR = "#FF0000"

    if robot == "iiwa14":
        fig, axes = plt.subplots(ncols=2, figsize=(6, 6))
    else:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    robot_constraints = JOINT_CONSTRAINTS[robot]

    plot_constraints = ["torque", "joint_velocity_dps"]
    constraint_names = ["Torque [Nm]", "Speed [deg/s]"]

    for i, constraint in enumerate(plot_constraints):
        col_labels = [constraint_names[i]]
        constraint_data = [robot_constraints[constraint][str(joint + 1)] for joint in range(NUM_JOINTS)]
        text_data = [constraint_data]
        if robot != "iiwa14":
            # Add a column to compare the joint constraints
            col_labels.append("vs iiwa14")
            iiwa14_constraint_data = [JOINT_CONSTRAINTS["iiwa14"][constraint][str(joint + 1)] for joint in range(NUM_JOINTS)]
            compared_data = [((constraint - iiwa14_constraint)/iiwa14_constraint)*100 for iiwa14_constraint, constraint in zip(iiwa14_constraint_data, constraint_data)]
            text_data.append(compared_data)

        text_data = np.array(text_data).T.astype(int)

        table = axes[i].table(cellText=text_data, colLabels=col_labels, cellLoc="center", loc="center", colColours=["#1F1F1F"]*len(col_labels), cellColours=[["#1F1F1F"]*len(col_labels)]*7)
        table.auto_set_font_size(False)
        table.set_fontsize(TABLE_COLUMN_FONT_SIZE)
        table.scale(1, 3)

        # For each cell in the table, set the text color
        for j, (row, col) in enumerate(table.get_celld().keys()):
            # First column, change all to white
            if col == 0 or row == 0:
                table[(row, col)].get_text().set_color(TEXT_COLOR)
                continue
            # Second column, if positive set green and add +, if negative set red
            if col == 1:
                if row == 0:
                    continue
                text = table[(row, col)].get_text()
                value = text.get_text()
                prefix = ""
                if value == "0":
                    text.set_color(TEXT_COLOR)
                elif value[0] == "-":
                    text.set_color(WORSE_COLOR)
                else:
                    text.set_color(IMPROVEMENT_COLOR)
                    prefix="+"
                text.set_text(f"{prefix}{value}%")
        
        axes[i].axis("off")

    fig.patch.set_facecolor(BACKGROUND_COLOR)
    fig.tight_layout()
    fig.savefig(f"{PACKAGE_ROOT}/../benchmarks/{robot}/joint_constraints.png")
    plt.close(fig)




if __name__ == "__main__":
    compare_joint_constraints("iiwa14")
    compare_joint_constraints("kr6r900")
    compare_joint_constraints("slugger")
    pass