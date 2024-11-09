import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import DEFAULT_TIMESTEP
from iiwa_batter.sandbox.pitch_check import make_model_directive


def loss_function():
    return 0


def run_instantaneous_swing(
    meshcat,
    plate_joint_positions,
    plate_joint_velocities,
    plate_ball_state_arrays: tuple[np.ndarray, np.ndarray],
    dt=DEFAULT_TIMESTEP,
):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(plate_joint_positions, dt)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Get the position of the sweet spot
    sweet_spot = plant.GetModelInstanceByName("sweet_spot")
    sweet_spot_position = plant.GetPositions(plant_context, sweet_spot)
    # If the sweet spot is too far from the ball's position when it crosses the strike zone, stop here and return cost function

    # The bat is in a position where it has a chance to hit the ball
    # Ensure there are no self-collisions. If there are, return some metric of how bad the self-collision is

    # Set the the ball's position and velocity at home plate, then execute the swing
    # We're starting at 0.45 seconds and going to 0.5 seconds
    ball = plant.GetModelInstanceByName("ball")
    plant.SetPositions(plant_context, ball, plate_ball_state_arrays[0])
    plant.SetVelocities(plant_context, ball, plate_ball_state_arrays[1])

    # Set the joint velocities of the iiwa
    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetVelocities(plant_context, iiwa, plate_joint_velocities)

    # Just turn off the torque for the time being
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, [0] * 7)

    # If the ball is traveling forwards, get how far it flies.
    if meshcat is not None:
        meshcat.StartRecording()

    simulator.AdvanceTo(0.05)

    if meshcat is not None:
        meshcat.PublishRecording()
