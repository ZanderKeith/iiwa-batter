import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from iiwa_batter import PACKAGE_ROOT, DEFAULT_TIMESTEP
from iiwa_batter.physics import feet_to_meters, mph_to_mps

def make_model_directive(initial_joint_positions, dt=DEFAULT_TIMESTEP):
    # We're pitchin the ball from +x to -x
    # The robot is sitting at the origin because I ain't messing with blender

    plate_offset_y = -0.8
    pitch_start_x = feet_to_meters(60.5) # Pitcher's mound is 60.5 feet from home plate
    pitch_start_z = feet_to_meters(5.9)

    strike_zone_z = 0.6

    model_directive = f"""
directives:
- add_model:
    name: iiwa
    file: file://{PACKAGE_ROOT}/assets/iiwa14.urdf
    default_joint_positions:
        iiwa_joint_1: [{initial_joint_positions[0]}]
        iiwa_joint_2: [{initial_joint_positions[1]}]
        iiwa_joint_3: [{initial_joint_positions[2]}]
        iiwa_joint_4: [{initial_joint_positions[3]}]
        iiwa_joint_5: [{initial_joint_positions[4]}]
        iiwa_joint_6: [{initial_joint_positions[5]}]
        iiwa_joint_7: [{initial_joint_positions[6]}]
- add_weld:
    parent: world
    child: iiwa::base
- add_model:
    name: bat
    file: file://{PACKAGE_ROOT}/assets/bat.sdf
- add_weld:
    parent: iiwa::iiwa_link_ee
    child: bat::base
    X_PC:
        translation: [0, 0, 0.2]
- add_model:
    name: sweet_spot
    file: file://{PACKAGE_ROOT}/assets/sweet_spot.sdf
- add_weld:
    parent: bat::base
    child: sweet_spot::base
    X_PC:
        translation: [0, 0, 0.4]
- add_model:
    name: ball
    file: file://{PACKAGE_ROOT}/assets/ball.sdf
    default_free_body_pose:
        base:
            translation: [{pitch_start_x}, {plate_offset_y}, {pitch_start_z}]
- add_model:
    name: strike_zone
    file: file://{PACKAGE_ROOT}/assets/strike_zone.sdf
- add_weld:
    parent: world
    child: strike_zone::base
    X_PC:
        translation: [0, {plate_offset_y}, {strike_zone_z}]

model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
      control_mode: torque_only

plant_config:
    contact_model: "hydroelastic"
    discrete_contact_approximation: "lagged"
    time_step: {dt}
    penetration_allowance: 8e-2
"""

    return model_directive

def run_pitch_check(meshcat, record_time, pitch_velocity_mph=90, save_time = dt=DEFAULT_TIMESTEP):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(joint_positions, dt)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    # Just turn off the torque for the time being
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, [0]*7)

    # Set initial velocity of the ball
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    ball = plant.GetModelInstanceByName("ball")
    ball_velocity_x = mph_to_mps(pitch_velocity_mph)
    plant.SetVelocities(plant_context, ball, np.array([0, 0, 0] + [-1*ball_velocity_x] + [0, -0.2]))


    if meshcat is not None:
        meshcat.StartRecording()

    times = np.linspace(0, record_time, int(record_time / dt))
    ball_x_positions = np.zeros_like(times)
    ball_x_velocities = np.zeros_like(times)
    for i, time in enumerate(times):
        simulator.AdvanceTo(time)
        ball_x_positions[i] = plant.GetPositions(plant_context, ball)[4]
        ball_x_velocities[i] = plant.GetVelocities(plant_context, ball)[3]

    if meshcat is not None:
        meshcat.PublishRecording()