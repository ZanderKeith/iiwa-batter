import os

from iiwa_batter import PACKAGE_ROOT

def export_iiwa_keyframes(state_dict, name):
    iiwa_frames = ["q1,q2,q3,q4,q5,q6,q7"]
    for time in state_dict.keys():
        iiwa_position = state_dict[time]["iiwa"][0]
        iiwa_frames.append(",".join([str(joint) for joint in iiwa_position]))

    with open(f"{PACKAGE_ROOT}/../blender/keyframes/{name}/iiwa.txt", "w") as f:
        f.write("\n".join(iiwa_frames))

def export_ball_keyframes(state_dict, name):
    ball_frames = ["x,y,z"]
    for time in state_dict.keys():
        ball_position = state_dict[time]["ball"][0]
        ball_frames.append(f"{ball_position[0]},{ball_position[1]},{ball_position[2]}")

    with open(f"{PACKAGE_ROOT}/../blender/keyframes/{name}/ball.txt", "w") as f:
        f.write("\n".join(ball_frames))

def export_keyframes(status_dict, name):
    if not os.path.exists(f"{PACKAGE_ROOT}/../blender/keyframes/{name}"):
        os.makedirs(f"{PACKAGE_ROOT}/../blender/keyframes/{name}")

    state_dict = status_dict["state"]
    export_ball_keyframes(state_dict, name)
    export_iiwa_keyframes(state_dict, name)

if __name__ == "__main__":
    # Export selected keyframes in a format that can be copy-pasted into Blender

    pass