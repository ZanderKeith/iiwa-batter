from importlib.resources import files

PACKAGE_ROOT = files("iiwa_batter")

CONTACT_TIMESTEP = 2e-5
PITCH_TIMESTEP = 1e-3
CONTROL_TIMESTEP = 50e-3
