import json
import numpy as np

from iiwa_batter import PACKAGE_ROOT

# Going off of this
#https://www.kuka.com/-/media/kuka-downloads/imported/87f2706ce77c4318877932fb36f6002d/kuka_robotrange_en.pdf?rev=2359724cb3a447a78aa636bfe3a0d5bf

# LBR iiwa 14 R820
# https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000246833_en.pdf?rev=e900f0e902d74aab863cdb4662512c74&hash=61E35585D464CAD421EE772689765FD4

# KR 6 R900 CR
# https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000293591_en.pdf?rev=dc9e6cb26ef7427580ab061bef95c0d8&hash=BE27A4044ECF975CCA1C1D7649CCDA27

# KR 8 R1440-2 arc HW
# https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000355820_en.pdf?rev=f5b24d4312264c1db9c68e18b70a3dba&hash=91561F39182A5B1DDC56207C38D44705

# In the future, might need to lock the fixed 

def lock_missing_joint(robot_constraints: dict) -> dict:
    for robot in ROBOT_CONSTRAINTS.keys():
        for joint in ROBOT_CONSTRAINTS[robot]["joint_range"].keys():
            if joint not in robot_constraints["joint_range"]:
                robot_constraints["joint_range"][joint] = ROBOT_CONSTRAINTS[robot]["joint_range"][joint]

def convert_to_radians(robot_constraints: dict) -> dict:
    for key in robot_constraints.keys():
        if key == "joint_range_deg":
            for joint, limits in robot_constraints[key].items():
                robot_constraints["joint_range"][joint] = np.deg2rad(limits)
        elif key == "joint_velocity_limits":
            for joint, limits in robot_constraints[key].items():
                robot_constraints[key][joint] =
        elif key == "joint_acceleration_limits":
            for joint, limits in robot_constraints[key].items():
                robot_constraints[key][joint] = [x * 3.14159 / 180 for x in limits]
    return robot_constraints

initial_robot_constraints = json.load(open(f"{PACKAGE_ROOT}/robot_constraints/kuka_details.json"))

ROBOT_CONSTRAINTS
