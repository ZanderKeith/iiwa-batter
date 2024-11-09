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

# In the future, might need to lock more joints... will cross that bridge when we get there

def lock_missing_joint(robot_constraints: dict) -> dict:
    lock_range = [0, 0]
    lock_velocity = 0
    new_dict = {}
    for robot, values in robot_constraints.items():
        # Get the joint that is skipped and kick all the other values down
        try:
            skipped_joint = int(values["skipped_joint"])
        except KeyError:
            new_dict[robot] = values
            continue
        
        # Copy old values over
        new_dict[robot] = {}
        for key in values.keys():
            if key not in ["joint_range_deg", "joint_velocity_dps"]:
                new_dict[robot][key] = values[key]
        new_dict[robot]["joint_range_deg"] = {}
        new_dict[robot]["joint_velocity_dps"] = {}

        existing_joints = [int(joint) for joint in values["joint_range_deg"].keys()]
        # Find the index of the joint that is skipped
        skipped_joint_index = existing_joints.index(skipped_joint)
        for joint in existing_joints[:skipped_joint_index]:
            joint_str = str(joint)
            new_dict[robot]["joint_range_deg"][joint_str] = values["joint_range_deg"][joint_str]
            new_dict[robot]["joint_velocity_dps"][joint_str] = values["joint_velocity_dps"][joint_str]
       
        new_dict[robot]["joint_range_deg"][str(skipped_joint)] = lock_range
        new_dict[robot]["joint_velocity_dps"][str(skipped_joint)] = lock_velocity

        for joint in existing_joints[skipped_joint_index:]:
            new_dict[robot]["joint_range_deg"][str(joint + 1)] = values["joint_range_deg"][str(joint)]
            new_dict[robot]["joint_velocity_dps"][str(joint + 1)] = values["joint_velocity_dps"][str(joint)]

    return new_dict

def convert_to_radians(robot_constraints: dict) -> dict:
    new_dict = {}

    for robot, values in robot_constraints.items():
        new_dict[robot] = values
        new_dict[robot]["joint_range"] = {}
        new_dict[robot]["joint_velocity"] = {}

        for joint, joint_values in values["joint_range_deg"].items():
            new_dict[robot]["joint_range"][joint] = [np.deg2rad(value) for value in joint_values]
        for joint, joint_value in values["joint_velocity_dps"].items():
            new_dict[robot]["joint_velocity"][joint] = np.deg2rad(joint_value)

    return new_dict

def scale_torque_limits(robot_constraints: dict) -> dict:
    target_robot = "iiwa14"
    original_torque_constraints = robot_constraints[target_robot]["torque"]

    new_dict = {}
    for robot, values in robot_constraints.items():
        if robot == target_robot:
            new_dict[robot] = values
            continue
        
        # More rated payload -> more torque
        payload_ratio = values["rated_payload"] / robot_constraints[target_robot]["rated_payload"]
        # More arm extension -> more torque
        extension_ratio = values["arm_extension"] / robot_constraints[target_robot]["arm_extension"]

        new_dict[robot] = values
        new_dict[robot]["torque"] = {}
        for joint in values["joint_range"].keys():
            new_dict[robot]["torque"][joint] = original_torque_constraints[joint] * payload_ratio * extension_ratio

    return robot_constraints

initial_robot_constraints = json.load(open(f"{PACKAGE_ROOT}/robot_constraints/kuka_details.json"))

skipped_joint_robot_constraints = lock_missing_joint(initial_robot_constraints)

converted_robot_constraints = convert_to_radians(skipped_joint_robot_constraints)

JOINT_CONSTRAINTS = scale_torque_limits(converted_robot_constraints)

print("Loaded joint constraints for KUKA robots")