import os

import dill

from iiwa_batter import PACKAGE_ROOT


def trajectory_dir(trajectory_settings, name):
    robot = trajectory_settings["robot"]
    pitch_speed_mph = trajectory_settings["pitch_speed_mph"]
    target_position_y = trajectory_settings["target_position"][1]
    target_position_z = trajectory_settings["target_position"][2]

    save_directory = f"{PACKAGE_ROOT}/swing_optimization/trajectories/{name}/{robot}_{pitch_speed_mph}mph_y{target_position_y}_z{target_position_z}"

    return save_directory


def save_trajectory(
    trajectory_settings,
    name,
    best_control_vector,
    best_reward,
    rewards,
    reward_differences,
):
    save_directory = trajectory_dir(trajectory_settings, name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(f"{save_directory}/best_trajectory.dill", "wb") as f:
        dill.dump((best_control_vector, best_reward, rewards, reward_differences), f)


def load_trajectory(trajectory_settings, name):
    """Load the best trajectory from a previous optimization run

    Parameters:
    ----------
    trajectory_settings: dict
        The settings used to generate the trajectory
    name: str
        The name of the optimization run

    Returns:
    -------
    tuple
        The best control vector, best reward, rewards, and reward differences
    """
    load_directory = trajectory_dir(trajectory_settings, name)
    with open(f"{load_directory}/best_trajectory.dill", "rb") as f:
        return dill.load(f)
