import numpy as np

from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS

# At certain steps of the process, include the human-created things as an initial guess

SWING_IMPACT = {
    "iiwa14": {
        "plate_position": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "plate_velocity": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
    }
}

COARSE_LINK = {
    "iiwa14": {
        "initial_position": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        # Wow we're really doing this, aren't we?
    }
}

def keyframe_trajectory(trajectory_timesteps, keyframes):
    # Given a list of keyframes, make a trajectory using rectilinear interpolation
    pass

def student_control_vector(robot, vector, type):
    
    joint_constraints = JOINT_CONSTRAINTS[robot]
    if type == "torque":
        torque_constraints = np.array([torque for torque in joint_constraints["torque"].values()])
        
    