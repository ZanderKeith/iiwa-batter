import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    Diagram,
    Simulator,
)

from iiwa_batter import CONTACT_TIMESTEP, PITCH_TIMESTEP
from iiwa_batter.physics import ball_flight_path
from iiwa_batter.sandbox.pitch_check import FLIGHT_TIME_MULTIPLE, make_model_directive

# We're just gonna quickfast see how slow this is


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def interpolate_trajectory(timesteps, trajectory):
    return {find_nearest(timesteps, time): phase for time, phase in trajectory.items()}

def setup_simulator(dt=CONTACT_TIMESTEP, meshcat=None):
    builder = DiagramBuilder()
    model_directive = make_model_directive(dt)
    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)

    return simulator, station

def reset_simulator(simulator: Simulator):
    context = simulator.get_mutable_context()
    context.SetTime(0)
    simulator.Initialize()


def run_full_trajectory(
    meshcat,
    simulator: Simulator,
    station: Diagram,
    initial_joint_positions,
    initial_ball_state_arrays,
    time_of_flight,
    robot_constraints,
    torque_trajectory: dict[float, np.ndarray],
):
    if meshcat is not None:
        meshcat.Delete()
    context = simulator.get_context()

    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, [0] * 7)

    simulator.AdvanceTo(0)
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Set the initial ball state
    ball = plant.GetModelInstanceByName("ball")
    plant.SetPositions(
        plant_context,
        ball,
        np.concatenate([np.ones(1), np.zeros(3), initial_ball_state_arrays[0]]),
    )
    plant.SetVelocities(
        plant_context, ball, np.concatenate([np.zeros(3), initial_ball_state_arrays[1]])
    )

    # Set initial iiwa state
    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(plant_context, iiwa, initial_joint_positions)
    plant.SetVelocities(plant_context, iiwa, [0] * 7)

    # Run the pitch

    if meshcat is not None:
        meshcat.StartRecording()

    simulation_duration = time_of_flight * FLIGHT_TIME_MULTIPLE
    timebase = np.arange(0, simulation_duration, PITCH_TIMESTEP)

    interpolated_trajectory = interpolate_trajectory(timebase, torque_trajectory)

    strike_distance = None
    for t in timebase:
        if t in interpolated_trajectory:
            station.GetInputPort("iiwa.torque").FixValue(
                station_context, interpolated_trajectory[t]
            )
        simulator.AdvanceTo(t)

        # Check for self-collision... eventually
        # station.GetOutputPort("contact_results").Eval(station_context)
        # contact_results = station.GetOutputPort("contact_results")

        # Enforce joint position and velocity limits... eventually
        # joint_positions = plant.GetPositions(plant_context, iiwa)

        # Record the distance between the sweet spot of the bat and the ball at the point when the ball passes through the strike zone
        # to include in our loss function
        if t > time_of_flight:
            if strike_distance is None:
                plant.HasModelInstanceNamed("sweet_spot")
                sweet_spot = plant.GetModelInstanceByName("sweet_spot")
                sweet_spot_body = plant.GetRigidBodyByName("base", sweet_spot)
                sweet_spot_pose = plant.EvalBodyPoseInWorld(
                    plant_context, sweet_spot_body
                )
                ball_position = plant.GetPositions(plant_context, ball)[4:]
                strike_distance = np.linalg.norm(
                    ball_position - sweet_spot_pose.translation()
                )

    if meshcat is not None:
        meshcat.PublishRecording()

    # Calculate reward
    # If ball position is negative, we missed the ball and should penalize
    # Otherwise, return reward based on the distance the ball travels

    ball_position = plant.GetPositions(plant_context, ball)[4:]

    if ball_position[0] < 0:
        reward = -10 * strike_distance
    else:
        # Determine distance ball travels
        ball_velocity = plant.GetVelocities(plant_context, ball)[3:]
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)  # Absolute distance from origin
        # If the ball is traveling backwards, reward is negative distance
        if land_location[0] < 0:
            reward = -distance
        # If the ball is foul (angle > +/- 45 degrees), reward is half the distance
        elif np.abs(np.arctan(land_location[1] / land_location[0])) > np.pi / 4:
            reward = distance / 2
        # Otherwise, return the distance
        else:
            reward = distance

    return reward
