import dill
import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

# At certain steps of the process, include the human-created things as an initial guess

SWING_IMPACT = {
    "iiwa14": {
        "plate_position": np.array([0.75, 0.25, 0.01, -0.6, 0, 0.55, 0]),
        "plate_velocity": np.array([1, 0, 1, 0, -0.4, 0, -0.5]),
    }
}

with open(f"{PACKAGE_ROOT}/../trajectories/student/iiwa14_link.dill", "rb") as f:
    iiwa14_link_control_vector = dill.load(f)

COARSE_LINK = {
    "iiwa14": {
        "initial_position": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "control_vector": iiwa14_link_control_vector,
    }
}

def student_control_vector(robot, vector, type):
    joint_constraints = JOINT_CONSTRAINTS[robot]
    if type == "torque":
        torque_constraints = np.array([torque for torque in joint_constraints["torque"].values()])
        control_vector = torque_constraints * vector
    elif type == "velocity":
        velocity_constraints = np.array([velocity for velocity in joint_constraints["joint_velocity"].values()])
        control_vector = velocity_constraints * vector
    elif type == "position":
        position_lower_constraints = np.array([position[0] for position in joint_constraints["joint_range"].values()])
        position_upper_constraints = np.array([position[1] for position in joint_constraints["joint_range"].values()])
        chosen_direction = np.where(vector > 0, position_upper_constraints, position_lower_constraints)
        control_vector = chosen_direction * np.abs(vector)
    else:
        raise ValueError("Invalid control vector type")
    return control_vector


def keyframe_trajectory(robot, trajectory_timesteps, keyframes):
    # Given a list of keyframes, make a trajectory by repeating the last keyframe until the next keyframe
    trajectory = {}
    keys = sorted(keyframes.keys())

    for i, timestep in enumerate(trajectory_timesteps):
        if len(keys) == 0:
            trajectory[timestep] = trajectory[trajectory_timesteps[i-1]]
        elif timestep >= keys[0]:
            trajectory[timestep] = keyframes[keys[0]]
            keys.pop(0)
        else:
            trajectory[timestep] = trajectory[trajectory_timesteps[i-1]]

    torque_trajectory = {key: student_control_vector(robot, trajectory[key], "torque") for key in trajectory.keys()}

    return torque_trajectory


def trajectory_to_control_vector(robot, trajectory):
    control_vector = np.array([trajectory[key] for key in sorted(trajectory.keys())])
    with open(f"{PACKAGE_ROOT}/trajectories/student/{robot}_link.dill", "wb") as f:
        dill.dump(control_vector, f)

    