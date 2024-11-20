from importlib.resources import files

PACKAGE_ROOT = files("iiwa_batter")

CONTACT_TIMESTEP = 2e-5  # Required for accurate hydroelastic contact simulation
PITCH_TIMESTEP = 1e-3  # Required to accurately plot ball's flight path
CONTROL_TIMESTEP = 10e-3  # How often the control of the robot is updated. (got decent results with 50ms, gonna try this out)
