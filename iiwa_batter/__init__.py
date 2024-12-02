from importlib.resources import files

PACKAGE_ROOT = files("iiwa_batter")

CONTACT_DT = 2e-5  # Required for accurate hydroelastic contact simulation
PITCH_DT = 1e-4  # Required to accurately plot ball's flight path and be consistent with the contact dt
CONTROL_DT = 10e-3  # How often the control of the robot is updated. (got decent results with 50ms too)
BLENDER_DT = (1/60) # 60 fps

NUM_JOINTS = 7  # Everything has 7 joints, cool
