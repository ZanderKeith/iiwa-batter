import bpy
import csv
from mathutils import Euler
import numpy as np

# Dictionary that contains info on how each joint in the armature should rotate
# This converts from Drake rotation to Blender rotation
# (moreso just a conversion to this particular rig because I'm new to rigging and wasn't consistent)

ARMATURE_DICT = {
    "link_1": {
        "index": 0, # Index in the FK vector
        "axis": 1, # Index in the blender keyframe
        "inverted": False, # To flip the rotation direction so it matches Drake
    },
    "link_2": {
        "index": 1,
        "axis": 2,
        "inverted": True,
    },
    "link_3": {
        "index": 2,
        "axis": 1,
        "inverted": False,
    },
    "link_4": {
        "index": 3,
        "axis": 2,
        "inverted": False,
    },
    "link_5": {
        "index": 4,
        "axis": 1,
        "inverted": False,
    },
    "link_6": {
        "index": 5,
        "axis": 1,
        "inverted": False,
    },
    "link_7": {
        "index": 6,
        "axis": 1,
        "inverted": False,
    }
}

D = bpy.data
C = bpy.context

def apply_iiwa_trajectory(trajectory):
    # Get the armature named iiwa14
    iiwa_armature = D.objects["iiwa14"]
    C.view_layer.objects.active = iiwa_armature
    bpy.ops.object.mode_set(mode='POSE')

    # Iterate through each pose in the trajectory
    for pose in trajectory:
        frame = pose[0]
        joint_angles = pose[1]
        # Iterate through each bone in the armature
        for bone_key in ARMATURE_DICT.keys():
            # Get the bone details to put rotation on the right axis
            bone = iiwa_armature.pose.bones[bone_key]
            bone_details = ARMATURE_DICT[bone_key]
            axis = bone_details["axis"]
            angle = joint_angles[bone_details["index"]]
            if bone_details["inverted"]:
                angle = angle * -1
            euler_angles = [0, 0, 0]
            euler_angles[axis] = angle
            
            # Insert a keyframe for the bone's quaternion rotation
            quaternion = Euler(euler_angles, 'XYZ').to_quaternion()
            if bone.rotation_quaternion.dot(quaternion) < 0: # make quaternion go in shortest direction
                quaternion.negate()
            bone.rotation_quaternion = quaternion
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

def load_iiwa_trajectory(file_name):
    trajectory = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            frame = i
            angles = np.array([float(angle) for angle in row])
            trajectory.append((frame, angles))
    return trajectory

def simple_iiwa_trajectory():
    # Keyframes are an array of 7 angles in radians
    pose_frame_0 = np.array([0, 0, 0, 0, 0, 0, 0])
    #pose_frame_100 = np.array([1, 1, 1, 1, 1, 1, 1])
    pose_frame_100 = np.array([0, 0, 0, 0, 0, 0, 1])
    pose_frame_0_deg = np.rad2deg(pose_frame_0)
    pose_frame_100_deg = np.rad2deg(pose_frame_100)
    return [(0, pose_frame_0_deg), (100, pose_frame_100_deg)]

trajectory = load_iiwa_trajectory("iiwa_positions_log.csv")
apply_iiwa_trajectory(trajectory)