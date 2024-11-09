import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.utils import RenderDiagram
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import PACKAGE_ROOT, DEFAULT_TIMESTEP
from iiwa_batter.sandbox.pitch_check import make_model_directive

def loss_function():
    return 0

def run_instantaneous_swing(meshcat, initial_joint_positions, initial_joint_velocities, ball_info:dict, dt=DEFAULT_TIMESTEP):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(initial_joint_positions, dt)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    # Set the the ball's initial position and velocity
    # We're starting at 0.45 seconds and going to 0.5 seconds

    DISTANCE_THRESHOLD = 0.1
    # Determine the distance from the sweet spot to the ball. If too far, stop here and return the distance

    # Ball is close enough, ensure there are no self-collisions. If there are, return double the distance

    # Execute swing, if the ball is still traveling backwards, return the distance

    # If the ball is traveling forwards, get how far it flies.



