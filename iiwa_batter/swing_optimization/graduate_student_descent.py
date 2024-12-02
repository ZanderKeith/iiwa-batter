import dill
import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    NUM_JOINTS
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

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
        raise ValueError(f"Invalid control vector type: {type}")
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
    with open(f"{PACKAGE_ROOT}/../trajectories/student/{robot}_link.dill", "wb") as f:
        dill.dump(control_vector, f)
    return control_vector

    
# At certain steps of the process, include the human-created things as an initial guess

SWING_IMPACT = {
    "iiwa14": {
        "plate_position": student_control_vector("iiwa14", np.array([0.75, 0.25, 0.01, -0.6, 0, 0.55, 0]), "position"),
        "plate_velocity": student_control_vector("iiwa14", np.array([1, 0, 1, 0, -0.4, 0, -0.5]), "velocity"),
    },
    "kr6r900": {
        "plate_position": student_control_vector("kr6r900", np.array([0.75, 0.68, 0, -0.6, 0, 0.55, 0.01]), "position"),
        "plate_velocity": student_control_vector("kr6r900", np.array([0.8, -0.5, 0, 0, 0.5, 0.5, -0.8]), "velocity")
    },
    "slugger": {
        "plate_position": student_control_vector("slugger", np.array([0.72, 0.33, 0, -0.67, 0, 0.55, 0.1]), "position"),
        "plate_velocity": student_control_vector("slugger", np.array([0.8, -0.8, 0.8, 0.8, 0, 0, -0.8]), "velocity")
    }
}

COARSE_LINK = {
    "iiwa14": {
        "initial_position": student_control_vector("iiwa14", np.array([0.5, 0.25, 0.01, -0.6, 0, 0.55, 0]), "position"),
        "control_vector": np.zeros((3, NUM_JOINTS)),
    },
    "kr6r900": {
        "initial_position": student_control_vector("kr6r900", np.array([0.1, 0.8, 0.01, -0.7, 0, 0.55, 0.2]), "position"),
        "control_vector": np.zeros((3, NUM_JOINTS))
    },
    # torque_keyframe_controls = {
    # 0: np.array([1, -0.58, 0, 0.21, 0, -0.1, -0.81]),
    # ball_time_of_flight: np.zeros(NUM_JOINTS),
    "slugger": {
        "initial_position": student_control_vector("slugger", np.array([0.35, 0.1, -0.5, -1, 0, 0.8, 0.9]), "position"),
        "control_vector": np.zeros((3, NUM_JOINTS))
    }
    # torque_keyframe_controls = {
    # 0: np.array([1, 0, 1, 0.3, 0, 0, -0.1]),
    # 0.4: np.array([1, -1, 1, 1, 0, 0, -1]),
    # ball_time_of_flight: np.zeros(NUM_JOINTS),
}


try:
    for robot in COARSE_LINK.keys():
        file_name = f"{PACKAGE_ROOT}/../trajectories/student/{robot}_link.dill"
        with open(file_name, "rb") as f:
            COARSE_LINK[robot]["control_vector"] = dill.load(f)
except FileNotFoundError:
    print(f"File not created! {file_name}")

